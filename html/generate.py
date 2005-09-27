#!/usr/bin/env python
from wiki2html import wiki2html, make_narrow

html = '''
<html>
<body>
'''

text = open('tutorial.txt').read()
content = wiki2html(text)
content = make_narrow(content)
html += content

html += '''
<p>
<hr>
2005 <a href="mailto:martinxyz@gmx.ch">Martin Renold</a>
</body>
</html>
'''

open('tutorial.html', 'w').write(html)
