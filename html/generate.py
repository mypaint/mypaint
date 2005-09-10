#!/usr/bin/env python
import htmlhelper

html = '''
<html>
<body>
'''

text = open('tutorial.txt').read()
content = htmlhelper.wiki2html(text)
content = htmlhelper.make_narrow(content)
html += content

html += '''
<p>
<hr>
2005 <a href="mailto:martinxyz@gmx.ch">Martin Renold</a>
</body>
</html>
'''

open('tutorial.html', 'w').write(html)
