import re, sys, os
with open(r'c:\Users\rahul\Downloads\Agentic RAG\app.py','r',encoding='utf-8') as f:
    src = f.read()
tq = '"""'
start = src.find('APP_HTML = ' + tq) + len('APP_HTML = ' + tq)
end = src.find(tq, start)
block = src[start:end]
m = re.search(r'<script>([\s\S]*?)</script>', block)
js = m.group(1)
out = os.path.join(os.environ.get('TEMP', '.'), 'local_js.js')
with open(out,'w',encoding='utf-8') as f:
    f.write(js)
print(out, len(js), 'chars')
