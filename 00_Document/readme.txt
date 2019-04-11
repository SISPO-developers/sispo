********************************************************************************
** Aaltothesis package
** This file contains the list of changes made in the package followed by the
** list of contents and brief instructions on how to use this package.
********************************************************************************

** Comments for aaltothesis package version 3.20
# Changes from previous release
# 2018-09-21 Luis Costa
- aaltothesis.cls underwent a major rewrite. The macros from the etoobox
  package were used to replace \ifthenelse and the tools from the calc package. 
  When creating a plain pdf file, including metadata is now possible. Adding
  additional metadata fields for equivalecy with the PDF/A version required
  the use of the hyperxmp package along with hooks defined in etoobox. This
  need for the use of etoolbox prompted replacement of the existing if
  structures and logical tests with the etoolbox tools. Additionally, the
  package was tested for deprecated code using the nag package with the options
  l2tabu and orthodox. As a result, the setting up of the page layout dimensions
  was redone. The option a-3b was removed from the template options since it is
  not required (the PDF/A-3b standard cannot be used to make your thesis
  document).

- The thesis templates opinnaytepohja.tex and thesistemplate.tex were also
  corrected to replace deprecated macros and environments. the amsmath package
  is now included so as to replace the decprecated eqnarray environment with
  align.

- A minimal template for writing your thesis in Swedish has been created by
  Henrik Wallen. This can be useful for editing the Finnish and English
  templates, too.

- The graphics in the templates have been renewed. The line diagram (figure 1)
  is redrawn to fix PDF/A-1b and PDF/A-2b compatibity issues experience 
  earlier (see comments for version 3.1). Also, a MATLAB graph has been added,
  and the last figure, figure 2 earlier, a photograph, has been replaced.

** Comments for aaltothesis package version 3.10
# Changes from previous release
# 2018-04-24 Luis Costa
- PDF/A support was extended to allow creating PDF/A-2b and PDF/A-3b compliant
  files by providing the \documentclass option a-2b and a-3b. The earlier option
  pdfa was replaced by a-1b. Thus all the PDF/A functionality provided by the
  pdfx.sty package is now directly available in the thesis template without
  having to hack the aaltothesis.cls file.

- The graphics files distributed in this package were pdf and eps files only.
  Now the distribution also contains jpg and png files as well. This is to
  stress the fact that different validators (Acrobat Pro, PDF-XChanger and the
  free online validator at https://www.pdf-online.com/osa/validate.aspx) given
  different validation results. In the example document, using pdflatex, the
  PDF/A-1b file created successfully passes the check made by all three
  validators for the jpg, pdf and png graphics files. However, when making a
  PDF/A-2b compliant file, using file kuva1.pdf in the example document, the
  resulting file fails the compliancy test in Acrobat Pro but passes it in the
  two other validators. Nonetheless, the resulting PDF/A-2b file is acceptable
  for archiving.

- Additional error and compatibility checking added.

- Hyperref option pdfstartview=FitH removed. It broke conformity rules on some
  validators.

** Comments for aaltothesis package version 3.02
# Changes from previous release
# 2018-03-29 Luis Costa
- The abstract page code was reconstructed to allow the abstract to span more
  than one page.

** Comments for aaltothesis package version 3.01
# Changes from previous release
# 2017-10-06 Luis Costa
- A bug in the page numbering was fixed.

- A minor modification (to clean up the code) was made to the \thesisappendix
  macro. 

# Changes from previous version
# 2017-10-05 Luis Costa
- A user interface for adding/changing the copyright text has been added. In
  version 3.0, this text was generated automatically.

- A small change, recommended earlier this week by Aalto, was made to the
  master's thesis abstract page: Department was changed to Degree programme and
  Professorship to Major.

** Comments for aaltothesis package version 3.0
# Changes from previous package
# 2017-09-27 Luis Costa
- PDF/A1-b support added to aaltothesis.cls, which is now the default. The page
  layout can be chosen to be the traditional single-side hardcopy print layout,
  the twoside hardcopy print layout (not recommended), or the symmetric layout
  for online publishing (this layout is to be submitted to the library for
  online publishing). The template files opinnaytepohja.tex and
  thesistemplate.tex have been modified accordingly. The biggest change in the
  templates is in the abstract text interface, since the abstract text must now
  go to the metadata of the pdf file as well, and in the numbering of the pages.
  As before, the use of the macros is documented in the template files in the
  macros themselves and the comments.

- NOTE: As of version 3.0, the package requires version 1.5.8 of the pdfx.sty 
  style package. Version 1.5.84 was the newest version available on the day of
  release of this package.

- The design of the abstract page has been changed to comply with the new
  recommendations given by Aalto University.

- The page numbering is continuous, beginning from the cover page to the end.
  The abstract page shows the number of pages upto the appendix, if it exists, +
  the number of pages the appendix (or appendices) has.

- The copyright of this package is now changed to an MIT license allowing free
  use of the package (see the copyright text in the class and template files for
  details).

** Comments for aaltothesis package version 2.2
# Changes from previous package
# 2015-04-24 Luis Costa
- aaltothesis.cls has been fixed so that the default font is now Latin Modern.
  There were serious issues in conjunction with MiKTeX that now have been fixed.

** Comments for aaltothesis package version 2.1
# Changes from previous package
# 2015-04-10 Luis Costa

- aaltothesis.cls (version 2.1) has been modified so that now on the cover page
  the title comes first followed by the author's name.

** Comments for aaltothesis package version 2.0
# Changes from previous package
# 2015-01-16 Luis Costa and Perttu Puska
- The \code of masters thesis major is no longer required. Doesn't affect
  bachelors degree.

** Comments for aaltothesis package version 1.7
# Changes from previous package (version 1.7)
# 2014-09-15 Luis Costa
- All definitions and specifications in class file aaltothesis.cls

- Support for Swedish added. The package now supports English, Finnish and
  Swedish. All language specific code is now in one place in the class file.

- utf8 is now the default encoding for the input. Also supported are latin1
  (iso-latin 1) and ansinew.


********************************************************************************
** Package contents**
********************************************************************************

aaltothesis.cls (class definitions)
opinnaytepohja.tex (thesis template in Finnish)
thesistemplate.tex (thesis template in English (only the macro documentation))
kuva1.eps  (graphics file)
kuva1.jpg  (graphics file)
kuva1.pdf  (graphics file)
kuva1.png  (graphics file)
kuva2.eps  (graphics file)
kuva2.jpg  (graphics file)
kuva2.pdf  (graphics file)
kuva2.png  (graphics file)
readme.txt (this file)


********************************************************************************
** Using this package**
********************************************************************************

1. Download the aaltologo-package from Aalto-Latex wiki pages

https://wiki.aalto.fi/download/attachments/49383512/aaltologo.zip

and unzip it to your working directory.

2. unzip (or download all components listed above) the
aaltothesis package

3. Edit file opinnaytepohja.tex or thesistemplate.tex (see remarks below)

4. Compile the file

In Linux:
 # pdflatex opinnaytepohja.tex
 (first run)
 # pdflatex opinnaytepohja.tex
 (second and final run)

In MikTeX (and/or the various GUIs):
 push the 'Build' or 'Build and View' button

********************************************************************************
** Remarks*
********************************************************************************

All changes are made to the file opinnaytepohja.tex or thesistemplate.tex.
There is no need to edit the file aaltothesis.cls. 

1. The default language in opinnaytepohja.tex is Finnish (and in 
   thesistemplate.tex it is English).

   To change this in opinnaytepohja.tex, uncomment 
    %\documentclass[english,12pt,...]{aaltothesis}
   and comment out
    \documentclass[finnish,12pt,...]{aaltothesis}

2. The default output format is now pdf/a-1b.

   To change this, leave out the a-1b option OR change
    \documentclass[...,a-1b,...]{aaltothesis}
   and uncomment 
    %\documentclass[...,dvips,...]{aaltothesis}
 
3. Choose the kind of thesis you will be creating.

   To change this, choose the correct degree by commenting and uncommenting 

    %\univdegree{BSc}
    \univdegree{MSc}
    %\univdegree{Lic} 

# End  
********************************************************************************
