from PIL import Image, ImageDraw, ImageFont
import json
import os
import xml.sax.saxutils
import argparse
from glob import glob
from tqdm import tqdm
from entities import ENTITY_ID_TO_NAME, NO_TEXT_ENTITIES, ENTITY_TABLE_COLUMN_ID, ENTITY_TABLE_ROW_ID, ENTITY_TABLE_ID, ENTITY_TABLE_HEADER_ROW_ID, ENTITY_TEXT_ID

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process document images and generate structured outputs.')
    parser.add_argument('--data-dir', type=str, default="data/436670b1e57c4add6af999be/annotated/CC-MAIN-2023-06/multimodal",
                        help='Directory containing the document images and annotation files')
    parser.add_argument('--filter', type=str, default=None,
                        help='Process only files matching this pattern (e.g., a specific page ID)')
    parser.add_argument('--draw-layout', action='store_true',
                        help='Draw layout annotations on the output images')
    return parser.parse_args()

def union(bbox1, bbox2):
    """Compute the union of two bounding boxes."""
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    xi1 = min(x1, x3)
    yi1 = min(y1, y3)
    xi2 = max(x2, x4)
    yi2 = max(y2, y4)

    return (xi1, yi1, xi2, yi2)

def to_xyxy(bbox):
    """Convert a bbox dict to (x1, y1, x2, y2) format."""
    x0 = bbox['x']
    y0 = bbox['y']
    x1 = bbox['x'] + bbox['width']
    y1 = bbox['y'] + bbox['height']
    return x0, y0, x1, y1

def iou(bbox1, bbox2):
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)

    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def is_all_inside(big_bbox, small_bbox):
    """Check if a small bounding box is inside a larger one."""
    offset_ratio = 0.01
    offset_x = int((big_bbox[2] - big_bbox[0]) * offset_ratio)
    offset_y = int((big_bbox[3] - big_bbox[1]) * offset_ratio)
    x1, y1, x2, y2 = big_bbox
    x3, y3, x4, y4 = small_bbox
    x1 = max(0, x1 - offset_x)
    y1 = max(0, y1 - offset_y)
    x2 = x2 + offset_x
    y2 = y2 + offset_y

    return x1 <= x3 and y1 <= y3 and x2 >= x4 and y2 >= y4

def merge(words):
    """Merge a list of word objects into a single word object."""
    merged_word = words[0]
    words = sorted(words, key=lambda w: (w['bbox']['y'], w['bbox']['x']))
    height_offset = 0.7
    for word in words[1:]:
        cur_x0, cur_y0 = merged_word['bbox']['x'], merged_word['bbox']['y']
        cur_x1, cur_y1 = cur_x0 + merged_word['bbox']['width'], cur_y0 + merged_word['bbox']['height']
        new_x0, new_y0 = word['bbox']['x'], word['bbox']['y']
        new_x1, new_y1 = new_x0 + word['bbox']['width'], new_y0 + word['bbox']['height']
        is_new_line = abs(cur_y0 - new_y0) > height_offset * (cur_y1 - cur_y0)
        cur_x0 = min(cur_x0, new_x0)
        cur_y0 = min(cur_y0, new_y0)
        cur_x1 = max(cur_x1, new_x1)
        cur_y1 = max(cur_y1, new_y1)
        merged_word['bbox']['x'] = cur_x0
        merged_word['bbox']['y'] = cur_y0
        merged_word['bbox']['width'] = cur_x1 - cur_x0
        merged_word['bbox']['height'] = cur_y1 - cur_y0
        # merged_word['text'] += "\n" if is_new_line else " "
        merged_word['text'] += " "
        merged_word['text'] += word['text']
        merged_word['entity_ids'] += word['entity_ids']
        merged_word['entity_categories'] += word['entity_categories']
    merged_word['text'] = merged_word['text'].strip()
    return merged_word

def find_text_by_id(lay_id, words, cat_id):
    """Find text associated with a layout ID."""
    inside_words = []
    for w in words:
        lay_index = w['entity_ids'].index(lay_id) if lay_id in w['entity_ids'] else - 1
        lay_cat_id = w['entity_categories'][lay_index] if lay_index != -1 else - 1
        if lay_cat_id == cat_id:
            inside_words.append(w)
    return merge(inside_words) if len(inside_words) > 0 else None

def read(path):
    """Read a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def draw(image, bboxes, text, color="red", width=2):
    """Draw bounding boxes and text on an image."""
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        draw = ImageDraw.Draw(image)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        draw.text((x0, y0), text, fill="red")
    return image

def find_el_id_from_cell(cell, target_ids):
    """Find element IDs from a cell that match target IDs."""
    el_ids = []
    for cat_id, ent_id in zip(cell['entity_categories'], cell['entity_ids']):
        if cat_id in target_ids and ent_id not in el_ids:
            el_ids.append(ent_id)
    return el_ids if len(el_ids) > 0 else []

def find_el_by_id(el_id, elements):
    """Find an element by its ID."""
    for el in elements:
        if el['id'] == el_id:
            return el
    return None

def sort_rows(rows_data, rows):
    """Sort table rows and cells within rows."""
    table = []
    rows_data = [[r_id, r] for r_id, r in rows_data.items()]
    rows_data = sorted(rows_data, key=lambda r: find_el_by_id(r[0], rows)['bbox'][1])
    rows_data = [r[1] for r in rows_data]
    for i, row in enumerate(rows_data):
        rows_data[i] = sorted(row, key=lambda r: r['bbox'][0])
    return rows_data

def rows_to_html_table(rows):
    """Convert row data to HTML table representation."""
    if not rows or not all(rows):
        return '<table border="1"></table>'
    
    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    
    for i in range(n_rows):
        if len(rows[i]) < n_cols:
            rows[i] = rows[i] + [{'id': f'empty_{i}_{j}', 'text': ''} for j in range(len(rows[i]), n_cols)]
    
    # Track cells that should be skipped (they're part of a span)
    skip_cells = [[False] * n_cols for _ in range(n_rows)]
    
    html_lines = ['<table border="1">']

    for i in range(n_rows):
        tag = 'td'  
        html_lines.append('  <tr>')
        
        for j in range(n_cols):
            if skip_cells[i][j]:
                continue

            cell = rows[i][j]
            cell_id = cell['id']
            cell_text = cell['text'].replace('\n', ' ')
            colspan = 1
            for k in range(j + 1, n_cols):
                # Only include adjacent cells with same ID that aren't already skipped
                if k < len(rows[i]) and not skip_cells[i][k] and rows[i][k]['id'] == cell_id:
                    colspan += 1
                    skip_cells[i][k] = True
                else:
                    break

            # Calculate rowspan (cells below with same ID)
            rowspan = 1
            for k in range(i + 1, n_rows):
                # Check if this position would create a valid span
                # It must have the same ID and not be already marked for skipping
                if j < len(rows[k]) and not skip_cells[k][j] and rows[k][j]['id'] == cell_id:
                    # Additionally, check that the entire span width is the same ID
                    # This ensures we don't create overlapping spans
                    valid_span = True
                    for col_offset in range(colspan):
                        col_idx = j + col_offset
                        if (col_idx >= len(rows[k]) or 
                            skip_cells[k][col_idx] or 
                            rows[k][col_idx]['id'] != cell_id):
                            valid_span = False
                            break
                    
                    if valid_span:
                        rowspan += 1
                        # Mark all cells in this span for skipping
                        for col_offset in range(colspan):
                            skip_cells[k][j + col_offset] = True
                    else:
                        break
                else:
                    break

            # Build the HTML attributes for the cell
            attrs = []
            if rowspan > 1:
                attrs.append(f'rowspan="{rowspan}"')
            if colspan > 1:
                attrs.append(f'colspan="{colspan}"')
            attr_str = ' ' + ' '.join(attrs) if attrs else ''

            html_lines.append(f'    <{tag}{attr_str}>{cell_text}</{tag}>')
        
        html_lines.append('  </tr>')
    html_lines.append('</table>')

    return '\n'.join(html_lines)

def rows_to_otsl_table(rows):
    """Convert row data to OTSL table format as per the paper specifications."""
    if not rows or not all(rows):
        return '<otsl></otsl>'
    
    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    
    # Create a grid to track cell spans
    grid = [['empty' for _ in range(n_cols)] for _ in range(n_rows)]
    
    # Fill grid with cell IDs to identify spans
    for i in range(n_rows):
        for j in range(min(len(rows[i]), n_cols)):
            grid[i][j] = rows[i][j]['id']
    
    otsl_lines = ['<otsl>']
    
    for i in range(n_rows):
        row_tokens = []
        for j in range(n_cols):
            if j >= len(rows[i]):  # Handle shorter rows
                cell_id = f'empty_{i}_{j}'
                cell_text = ''
            else:
                cell = rows[i][j]
                cell_id = cell['id']
                cell_text = cell['text'].replace('\n', ' ')
            
            # Determine cell type based on spans
            if i > 0 and j > 0 and grid[i][j] == grid[i-1][j] and grid[i][j] == grid[i][j-1]:
                # Cross cell - spans from both above and left
                row_tokens.append('X')
            elif i > 0 and grid[i][j] == grid[i-1][j]:
                # Up-looking cell - spans from above
                row_tokens.append('U')
            elif j > 0 and grid[i][j] == grid[i][j-1]:
                # Left-looking cell - spans from left
                row_tokens.append('L')
            else:
                # Content cell - either new cell or span origin
                row_tokens.append(f'C {cell_text}' if cell_text else 'C')
        
        otsl_lines.append(''.join(row_tokens) + 'NL')
    
    otsl_lines.append('</otsl>')
    return '\n'.join(otsl_lines)

def find_annot_el_by_id(lay_id, annot):
    """Find annotation element by ID."""
    for e, m in annot['entities'].items():
        for lay in m:
            if lay['id'] == lay_id:
                return lay
    return None

def is_duplicated(bboxes, bbox, iou_thr=0.9):
    """Check if a bounding box is duplicated."""
    for b in bboxes:
        if iou(b, bbox) > iou_thr:
            return True
    return False

def find_table_elements_by_id(table, annot, entity_ids):
    """Find table elements matching entity IDs."""
    found_bboxes = []
    table_bbox = to_xyxy(table['bbox'])
    lays = []
    for entity_id in entity_ids:
        lays += annot['entities'].get(str(entity_id), [])
    elements = []
    for lay in lays:
        lay_bbox = (lay['bbox']['x'], lay['bbox']['y'],
                   lay['bbox']['x'] + lay['bbox']['width'],
                   lay['bbox']['y'] + lay['bbox']['height'])
        if is_all_inside(table_bbox, lay_bbox) and not is_duplicated(found_bboxes, lay_bbox):
            x0, y0, x1, y1 = lay_bbox
            elements.append({
                "bbox": (x0, y0, x1, y1),
                "id": lay['id'],
            })
            found_bboxes.append(lay_bbox)
    return elements

def find_columns_by_table_id(table_id, annot):
    """Find columns for a table."""
    return find_table_elements_by_id(table_id, annot, [ENTITY_TABLE_COLUMN_ID])

def find_rows_by_table_id(table_id, annot):
    """Find rows for a table."""
    return find_table_elements_by_id(table_id, annot, [ENTITY_TABLE_ROW_ID, ENTITY_TABLE_HEADER_ROW_ID])

def escape_xml(text):
    """Escape XML special characters."""
    return xml.sax.saxutils.escape(str(text) if text is not None else "")

def elements_to_doctag_html(elements):
    """Convert elements to doctag format with HTML tables."""
    doctag_lines = ["<doctag>"]
    
    for el in elements:
        el_type = el.get("type", "unknown")
        el_text = el.get("text", "")  # HTML for tables, text for others
        
        if el_type == "table":
            # For tables, directly include the HTML content
            doctag_lines.append(f"<{el_type}>{el_text}</{el_type}>")
        elif el_type.startswith("heading_"):
            try:
                level = el_type.split("_")[-1]
                tag = f"section_header_level_{level}"
                escaped_text = escape_xml(el_text)
                doctag_lines.append(f"<{tag}>{escaped_text}</{tag}>")
            except:
                escaped_text = escape_xml(el_text)
                doctag_lines.append(f"<section_header>{escaped_text}</section_header>")
        else:
            escaped_text = escape_xml(el_text)
            doctag_lines.append(f"<{el_type}>{escaped_text}</{el_type}>")
    
    doctag_lines.append("</doctag>")
    return "\n".join(doctag_lines)

def elements_to_doctag_otsl(elements):
    """Convert elements to doctag format with OTSL tables."""
    doctag_lines = ["<doctag>"]
    
    for el in elements:
        el_type = el.get("type", "unknown")
        el_text = el.get("text", "")
        el_data = el.get("data", None)
        
        if el_type == "table" and el_data is not None:
            # Generate OTSL for tables
            otsl_content = rows_to_otsl_table(el_data)
            doctag_lines.append(otsl_content)
        elif el_type.startswith("heading_"):
            try:
                level = el_type.split("_")[-1]
                tag = f"section_header_level_{level}"
                escaped_text = escape_xml(el_text)
                doctag_lines.append(f"<{tag}>{escaped_text}</{tag}>")
            except:
                escaped_text = escape_xml(el_text)
                doctag_lines.append(f"<section_header>{escaped_text}</section_header>")
        else:
            escaped_text = escape_xml(el_text)
            doctag_lines.append(f"<{el_type}>{escaped_text}</{el_type}>")
    
    doctag_lines.append("</doctag>")
    return "\n".join(doctag_lines)

def deduplicate_bboxes(cells, iou_thr=0.9):
    """Remove duplicate bounding boxes."""
    deduplicated_cells = []
    seen_bboxes = set()
    for cell in cells:
        bbox = tuple(cell['bbox'])
        if not is_duplicated(seen_bboxes, bbox, iou_thr):
            deduplicated_cells.append(cell)
            seen_bboxes.add(bbox)
    return deduplicated_cells

def main():
    """Main entry point."""
    args = parse_args()
    
    # Find all document images in the data directory
    page_pattern = os.path.join(args.data_dir, "doc_*.jpg")
    pages = glob(page_pattern)
    pages = sorted(pages) 
    
    # Create output directories
    output_dirs = {
        "layouts": "output/layouts",
        "markdown": "output/markdown_outputs",
        "doctag_html": "output/doctag_html_outputs",
        "doctag_otsl": "output/doctag_otsl_outputs",
        "languages": "output/languages",
    }
    
    for dir_name in output_dirs.values():
        os.makedirs(dir_name, exist_ok=True)

    
    # Process each page
    for page_image in tqdm(pages, total=len(pages), desc="Processing pages"):
        # Extract page name and paths to annotation files
        page_name = os.path.basename(page_image).split(".")[0]
        
        # Skip if filter is provided and doesn't match
        if args.filter and args.filter not in page_name:
            continue
            
        layout_annot_path = page_image.replace("/doc_", "/entities_doc_").replace(".jpg", ".json")
        text_annot_path = page_image.replace("/doc_", "/words_doc_").replace(".jpg", ".json")
        
        # Load annotations and image
        try:
            layout_annot = read(layout_annot_path)
            text_annot = read(text_annot_path)
            image = Image.open(page_image)
        except Exception as e:
            print(f"Error loading {page_image}: {e}")
            continue
        # Initialize data structures
        tables = {}
        astray_cells = []
        elements = []
        page_languages = text_annot.get('metadata', {}).get('languages_fasttext', {'__label__unknown': 1})
        page_top_language = max(page_languages, key=page_languages.get)
        page_language = page_top_language.split('__label__')[-1]
        page_language_conf = page_languages[page_top_language]
        if page_language == 'unknown':
            continue
        # Process layout annotations
        for ent, meta in layout_annot['entities'].items():
            for lay in meta:
                x0, y0, x1, y1 = to_xyxy(lay['bbox'])
                
                entity = ENTITY_ID_TO_NAME[int(ent)]
                lay_id = lay['id']
                
                # Draw layout on image if enabled
                if args.draw_layout:
                    image = draw(image, [(x0, y0, x1, y1)], entity)
                
                # Skip entities without text
                if int(ent) in NO_TEXT_ENTITIES:
                    continue
                    
                # Get text for this entity
                word = find_text_by_id(lay_id, text_annot['words'], int(ent))
                text = word['text'] if word else ""
                
                # Process table cells
                if entity in ['table_cell', 'table_header_cell']:
                    if not word:
                        astray_cells.append({
                            "bbox": (x0, y0, x1, y1),
                            "text": "",
                            "id": lay_id,
                            "col_ids": [],
                            "row_ids": [],
                            "is_header": entity == 'table_header_cell',
                        })
                        continue
                        
                    # Find the table this cell belongs to
                    table_id = find_el_id_from_cell(word, [ENTITY_TABLE_ID])
                    if not table_id:
                        continue
                        
                    table_id = table_id[0]
                    table = find_annot_el_by_id(table_id, layout_annot)
                    
                    # Create table entry if it doesn't exist
                    if table_id not in tables:
                        table_bbox = to_xyxy(table['bbox'])
                        tables[table_id] = {
                            "cells": [],
                            "columns": find_columns_by_table_id(table, layout_annot),
                            "rows": find_rows_by_table_id(table, layout_annot),
                            "bbox": table_bbox,
                            "id": table_id,
                            "has_header": False,
                        }
                    
                    # Add cell to table
                    cell = {
                        "bbox": (x0, y0, x1, y1),
                        "text": text,
                        "id": lay_id,
                        "col_ids": [],
                        "row_ids": [],
                        "is_header": entity == 'table_header_cell',
                    }
                    tables[table_id]['cells'].append(cell)
                    tables[table_id]['has_header'] = tables[table_id]['has_header'] or cell['is_header']
                
                # Process other types of content
                else:
                    if "head" in entity and ENTITY_TEXT_ID in word.get('entity_categories', []):
                        continue
                    elements.append({
                        "bbox": (x0, y0, x1, y1),
                        "text": text,
                        "id": lay_id,
                        "type": entity,
                    })
        
        # Process tables
        for table_id, table_data in tables.items():
            # print(f"Got {len(table_data['rows'])} rows and {len(table_data['columns'])} columns")
            tables[table_id]["rows_data"] = {}
            table_bbox = table_data['bbox']
            
            # Add any astray cells that belong in this table
            for astray_cell in astray_cells:
                cell_bbox = astray_cell['bbox']
                if is_all_inside(table_bbox, cell_bbox):
                    table_data['cells'].append(astray_cell)
            
            # Create header row if table has headers
            if table_data['has_header'] and table_data['rows']:
                tot_rows_bbox = table_data['rows'][0]['bbox']
                for row in table_data['rows']:
                    tot_rows_bbox = union(tot_rows_bbox, row['bbox'])
                table_width = table_bbox[2] - table_bbox[0]
                header_row_bbox = (table_bbox[0], table_bbox[1], tot_rows_bbox[0] + table_width, tot_rows_bbox[1])
                header_row = {"bbox": header_row_bbox, "id": table_id + "_header_row"}
                tables[table_id]['rows'].append(header_row)
                table_data['rows'].append(header_row)
            
            # Deduplicate cells
            table_data['cells'] = deduplicate_bboxes(table_data['cells'])
            
            # Assign cells to rows and columns
            num_cells = 0
            threshold = 0.5
            
            for i, cell in enumerate(table_data['cells']):
                # Map cells to columns
                for col in table_data['columns']:
                    inter_x0 = max(cell['bbox'][0], col['bbox'][0])
                    inter_x1 = min(cell['bbox'][2], col['bbox'][2])
                    
                    if inter_x0 >= inter_x1:
                        continue
                        
                    inter_width = inter_x1 - inter_x0
                    col_width = col['bbox'][2] - col['bbox'][0]
                    
                    if inter_width / col_width > threshold:
                        tables[table_id]['cells'][i]['col_ids'].append(col['id'])
                
                # Map cells to rows
                for row in table_data['rows']:
                    inter_y0 = max(cell['bbox'][1], row['bbox'][1])
                    inter_y1 = min(cell['bbox'][3], row['bbox'][3])
                    
                    if inter_y0 >= inter_y1:
                        continue
                        
                    inter_height = inter_y1 - inter_y0
                    row_height = row['bbox'][3] - row['bbox'][1]
                    
                    if inter_height / row_height > threshold:
                        tables[table_id]['cells'][i]['row_ids'].append(row['id'])
                
                # Remove duplicates
                cell['col_ids'] = list(set(cell['col_ids']))
                cell['row_ids'] = list(set(cell['row_ids']))
                for row_id in cell['row_ids']:
                    if row_id not in table_data['rows_data']:
                        table_data['rows_data'][row_id] = []
                    for col_id in cell['col_ids']:
                        table_data['rows_data'][row_id].append(cell)
                        num_cells += 1
            
            # print(f"Got {num_cells} cells")
            try:
                rows = sort_rows(table_data['rows_data'], table_data['rows'])
            except Exception as e:
                print(f"Error sorting rows: {e}")
                continue
            html_table = rows_to_html_table(rows)
            elements.append({
                "bbox": table_data['bbox'],
                "text": html_table,
                "data": rows,  # Store structured data for OTSL generation
                "id": table_id,
                "type": "table",
            })
            
            # print(html_table)

        elements = sorted(elements, key=lambda x: (x['bbox'][1], x['bbox'][0]))
        md = ""
        for el in elements:
            md += f"\n\n{el['text']}"
        md = md.strip()
        doctag_html = elements_to_doctag_html(elements)
        doctag_otsl = elements_to_doctag_otsl(elements)
        
        with open(os.path.join(output_dirs["markdown"], f"{page_name}.md"), "w", encoding="utf-8") as f:
            f.write(md)
            
        with open(os.path.join(output_dirs["doctag_html"], f"{page_name}.dt.xml"), "w", encoding="utf-8") as f:
            f.write(doctag_html)
            
        with open(os.path.join(output_dirs["doctag_otsl"], f"{page_name}.dt.xml"), "w", encoding="utf-8") as f:
            f.write(doctag_otsl)

        with open(os.path.join(output_dirs["languages"], f"{page_name}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "language": page_language,
                "confidence": page_language_conf
            }, f, indent=4)
        image.save(os.path.join(output_dirs["layouts"], f"{page_name}.jpg"))
        print("Language:", page_language)
        # print(f"{'#' * 10} {page_name} {'#' * 10}")

if __name__ == "__main__":
    main()
