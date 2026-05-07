#!/usr/bin/env python3
"""
Create Excel file from CSV files using zipfile and XML.
This creates a basic .xlsx file without requiring openpyxl or xlsxwriter.
"""

import os
import zipfile
import csv
from pathlib import Path


def csv_to_xml_sheet(csv_file, sheet_name):
    """Convert CSV to Excel XML sheet format."""
    rows = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    
    xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    xml += '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n'
    xml += '<sheetData>\n'
    
    for i, row in enumerate(rows, 1):
        xml += f'<row r="{i}">\n'
        for j, cell in enumerate(row, 1):
            col_letter = chr(64 + j) if j <= 26 else chr(64 + (j-1)//26) + chr(64 + ((j-1)%26 + 1))
            # Escape XML special characters
            cell_value = str(cell).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            xml += f'<c r="{col_letter}{i}" t="inlineStr"><is><t>{cell_value}</t></is></c>\n'
        xml += '</row>\n'
    
    xml += '</sheetData>\n'
    xml += '</worksheet>'
    
    return xml


def create_excel_from_csv(csv_dir, output_file):
    """Create Excel file from CSV files."""
    # Create Excel file structure
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Create [Content_Types].xml
        content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/worksheets/sheet3.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/worksheets/sheet4.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/worksheets/sheet5.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>'''
        zf.writestr('[Content_Types].xml', content_types)
        
        # Create _rels/.rels
        rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''
        zf.writestr('_rels/.rels', rels)
        
        # Create xl/_rels/workbook.xml.rels
        workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet3.xml"/>
<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet4.xml"/>
<Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet5.xml"/>
</Relationships>'''
        zf.writestr('xl/_rels/workbook.xml.rels', workbook_rels)
        
        # Create sheets
        sheets = [
            ('Summary.csv', 'Summary'),
            ('R2_gt_0.1.csv', 'R2_gt_0.1'),
            ('R2_gt_0.2.csv', 'R2_gt_0.2'),
            ('R2_gt_0.3.csv', 'R2_gt_0.3'),
            ('All_Data.csv', 'All_Data')
        ]
        
        sheet_names = []
        for i, (csv_file, sheet_name) in enumerate(sheets, 1):
            csv_path = os.path.join(csv_dir, csv_file)
            if os.path.exists(csv_path):
                xml = csv_to_xml_sheet(csv_path, sheet_name)
                zf.writestr(f'xl/worksheets/sheet{i}.xml', xml)
                sheet_names.append((i, sheet_name))
        
        # Create xl/workbook.xml
        workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets>'''
        for i, name in sheet_names:
            workbook += f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>'
        workbook += '''</sheets>
</workbook>'''
        zf.writestr('xl/workbook.xml', workbook)


if __name__ == '__main__':
    csv_dir = 'reports/output_20260203_145601/r2_statistics_summary_csv'
    output_file = 'reports/output_20260203_145601/r2_statistics_summary.xlsx'
    
    print("Creating Excel file from CSV files...")
    create_excel_from_csv(csv_dir, output_file)
    print(f"Excel file created: {output_file}")
