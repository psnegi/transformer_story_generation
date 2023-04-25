import os
import requests


ebook_nums = [11, 45, 120, 46, 55, 514, 16, 236, 41, 902, 17396, 12, 2781,  289, 113, 271, 421]

start_string = '*** START OF THE PROJECT GUTENBERG EBOOK'

end_string = '*** END OF THE PROJECT GUTENBERG'

file = "input.txt"
with open(file, 'w') as f:
    for en in ebook_nums:
        url = f"https://www.gutenberg.org/cache/epub/{en}/pg{en}.txt"
        r= requests.get(url, allow_redirects=True)
        #print(r.text)
        #print(type(r.text))
        idxs = r.text.find(start_string)    
        idxe = r.text.find(end_string)
        f.write(r.text[idxs+len(start_string) + 30: idxe])

