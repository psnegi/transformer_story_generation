import os
import requests
import gutenbergpy.textget

ebook_nums = [11, 45, 120, 46, 55, 514, 16, 236, 41, 902, 17396, 12, 2781,  289, 113, 271, 421]

start_string = '*** START OF THE PROJECT GUTENBERG EBOOK'

end_string = '*** END OF THE PROJECT GUTENBERG'

file = "input.txt"
with open(file, 'w') as f:
    for en in ebook_nums:
            # This gets a book by its gutenberg id number
        raw_book = gutenbergpy.textget.get_text_by_id(2701) # with headers
        clean_book = gutenbergpy.textget.strip_headers(raw_book)
        f.write(clean_book.decode())

