import os
import numpy

f = open("tables.txt","r")

lines = f.readlines()
table_strings = [""]

for line in lines:
    if line.isspace():
        table_strings.append("")
    else:
        table_strings[-1] = table_strings[-1] + line.strip()

tables = [eval(table_str) for table_str in table_strings]

html_str = ""


html_str = html_str+r"""<center><table border="1"><tr>"""
for row in tables[0]:
    html_str = html_str + "<tr>"
    for item in row:
        html_str = html_str+f"""<td align="middle"><font size="1">{item}</font></td>"""
    html_str = html_str + "</tr>"
html_str = html_str+r"""</table></center>"""
html_str = html_str+"\n\n\n"

p_id_to_arrow = {
    0:"&#8593;",
    1:"&#8595;",
    2:"&#8592;",
    3:"&#8594;"
}



html_str = html_str+r"""<center><table border="1"><tr>"""
for row in tables[1]:
    html_str = html_str + "<tr>"
    for item in row:
        html_str = html_str+f"""<td align="middle"><font size="1">{p_id_to_arrow[item]}</font></td>"""
    html_str = html_str + "</tr>"
html_str = html_str+r"""</table></center>"""

with open("tables.html","w") as file_out:
    file_out.write(html_str)
    file_out.flush()

