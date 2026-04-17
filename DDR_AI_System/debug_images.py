import sys; sys.path.insert(0,'.')
from pipeline.extractor import extract_document
from pipeline.report_builder import _build_image_map

doc = extract_document(r'..\Sample Report.pdf', 'inspection')
imap = _build_image_map(doc.images)
photo_keys = sorted([k for k in imap if k.startswith('Photo ')], key=lambda x: int(x.split()[1]))
print('Mapped photo count:', len(photo_keys))
print('First 10:', photo_keys[:10])
print('Last 5:', photo_keys[-5:])

refs = ['Photo 1','Photo 7','Photo 15','Photo 20','Photo 31','Photo 42','Photo 49','Photo 53','Photo 58','Photo 64']
for r in refs:
    found = r in imap
    print(r, ':', 'FOUND' if found else 'MISSING')
