#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:50:02 2024

@author: af
"""

from pdf2image import convert_from_path

# Path to your PDF file
pdf_path = 'your_document.pdf'

# Convert PDF to a list of images (one per page)
images = convert_from_path(pdf_path, dpi=300)  # dpi=300 for high-quality images

# Save each page as a PNG
for i, image in enumerate(images):
    image.save(f'page_{i + 1}.png', 'PNG')

print("PDF converted to PNG.")
