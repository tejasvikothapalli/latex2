import math
from data_management import *
string_numerizer = StringNumerizer('latex_vocab.txt')
# import
import re
from pdflatex import PDFLaTeX
#from tex import latex2pdf
template = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{amsmath}
\newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
\begin{document}
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""

def numlist2str(label_list):
    squence_str = ''
    strlist = []
    for i in range(len(label_list)):
        # print('hello')
        vocab_id = label_list[i]
        
        if vocab_id not in string_numerizer.idx2sym:
#            print(vocab_id)
            assert vocab_id in string_numerizer.idx2sym
        if vocab_id > 3 - 1:
            l = string_numerizer.idx2sym[vocab_id]
            
            # l = id2vocab[vocab_id-4]
            # if vocab_id == 4:
            #     l = 'UNK'
            # assert (l ~= nil, 'target vocab size incorrect!')
            
            strlist.append(l)
                
            
            squence_str = squence_str + l
            squence_str = squence_str + ' '
        
    label_str = squence_str
    return label_str, strlist

def get_latex(labels):
    target_l = len(labels)
    

    label_list = []
    for t in range(target_l):
        label = labels[t]
        if label == 3 - 1: #'<END>'
            break
        label_list.append(label)
        
    l, label_strlist = numlist2str(label_list)
    
    l = l.strip()
    l = l.replace(r'\pmatrix', r'\mypmatrix')
    l = l.replace(r'\matrix', r'\mymatrix')
    
    # remove leading comments
    l = l.strip('%')
    if len(l) == 0:
        l = '\\hspace{1cm}'
    # \hspace {1 . 5 cm} -> \hspace {1.5cm}
    for space in ["hspace", "vspace"]:
        match = re.finditer(space + " {(.*?)}", l)
        if match:
            new_l = ""
            last = 0
            for m in match:
                new_l = new_l + l[last:m.start(1)] + m.group(1).replace(" ", "")
                last = m.end(1)
            new_l = new_l + l[last:]
            l = new_l

    return l
    
#def get_picture(latex_markup):
#    with open('equation.tex', "w") as w:
#            w.write(latex_markup)
#            w.close()
#    pdfl = PDFLaTeX.from_texfile('equation.tex')
#    pdf, log, completed_process = pdfl.create_pdf(keep_pdf_file=True, keep_log_file=False)
