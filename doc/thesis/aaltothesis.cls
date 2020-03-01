%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% LaTeX document class 'aaltothesis', for use with pdflatex.
%% This is a class for typesetting theses to be submitted for examination to the
%% Aalto University School of Electrical Engineering, which may be extended for
%% use in other Aalto schools as well. This class file is designed to be used
%% with the template files opinnaytepohja.tex (Finnish) and thesistemplate.tex
%% (English). The template can be used in Swedish by saying "swedish" as the
%% language option in the \documentclass. This class enables the production of a
%% PDF/A-1b, PDF/A-2b or PDF/A-3b compliant file as made possible by the
%% pdfx.sty package.
%%
%% This class package requires pdfx.sty v. 1.5.84 (2017/05/18) or newer.
%%
%%
%% Written by Luis R.J. Costa.
%% Currently developed at the Learning Services of Aalto University School of
%% Electrical Engineering by Luis R.J. Costa since May 2017.
%%
%% This is licensed under the terms of the MIT license below.
%%
%% Copyright 2017-2018, by Luis R.J. Costa, luis.costa@aalto.fi,
%% Copyright 2017-2018 Swedish translations in aaltothesis.cls by Elisabeth
%% Nyberg, elisabeth.nyberg@aalto.fi and Henrik Wallén, henrik.wallen@aalto.fi.
%% Copyright 2017-2018 Finnish documentation in the template opinnatepohja.tex
%% by Perttu Puska, perttu.puska@aalto.fi, and Luis R.J. Costa.
%% Copyright 2018 English template thesistemplate.tex by Luis R.J. Costa.
%% Copyright 2018 Swedish template kandidatarbetsbotten.tex by Henrik Wallen.
%%
%% Permission is hereby granted, free of charge, to any person obtaining a copy
%% of this software and associated documentation files (the "Software"), to deal
%% in the Software without restriction, including without limitation the rights
%% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
%% copies of the Software, and to permit persons to whom the Software is
%% furnished to do so, subject to the following conditions:
%% The above copyright notice and this permission notice shall be included in
%% all copies or substantial portions of the Software.
%% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
%% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
%% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
%% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
%% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
%% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
%% SOFTWARE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
%% Adding a new language:
%% 1. Add the language option using \DeclareOption as done a few lines below.
%%    It's assumed your language is supported by the babel package.
%% 2. Add your language to the options passed to babel
%% 3. Make a corresponding short-hand macro \IfYourlang (under "Some shorthand")
%% 4. Add the translations to the macros \setthesismainlangnames and
%%    \setabstractlangnames.
%% 5. Make the corresponding additions to the macros \L@ng and \settocabslang.
%% 6. Make your language option available in the abstractpage environment.
%%
%%
\NeedsTeXFormat{LaTeX2e}%
\ProvidesClass{aaltothesis}[2018/08/31 Aalto Univ. ELEC thesis class v3.20]%
\RequirePackage{etoolbox}% Defines boolean/toggle variables, ifs, AtEndPreamble..
%
\newcommand*{\MainLang}{}%
\DeclareOption{english}{%
  \AtEndOfClass{\main@language{english}}%
  \renewcommand*{\MainLang}{english}%
}%
\DeclareOption{finnish}{%
  \AtEndOfClass{\main@language{finnish}}%
  \renewcommand*{\MainLang}{finnish}%
}%
\DeclareOption{swedish}{%
  \AtEndOfClass{\main@language{swedish}}%
  \renewcommand*{\MainLang}{swedish}%
}%
%
\newtoggle{ARTS}%
\newtoggle{BIZ}%
\newtoggle{CHEM}%
\newtoggle{ELEC}%
\newtoggle{ENG}%
\newtoggle{SCI}%
\DeclareOption{arts}{%
  \toggletrue{ARTS}\togglefalse{BIZ}\togglefalse{CHEM}%
  \togglefalse{ELEC}\togglefalse{ENG}\togglefalse{SCI}%
}%
\DeclareOption{biz}{%
  \togglefalse{ARTS}\toggletrue{BIZ}\togglefalse{CHEM}%
  \togglefalse{ELEC}\togglefalse{ENG}\togglefalse{SCI}%
}%
\DeclareOption{chem}{%
  \togglefalse{ARTS}\togglefalse{BIZ}\toggletrue{CHEM}%
  \togglefalse{ELEC}\togglefalse{ENG}\togglefalse{SCI}%
}%
\DeclareOption{elec}{%
  \togglefalse{ARTS}\togglefalse{BIZ}\togglefalse{CHEM}%
  \toggletrue{ELEC}\togglefalse{ENG}\togglefalse{SCI}%
}%
\DeclareOption{eng}{%
  \togglefalse{ARTS}\togglefalse{BIZ}\togglefalse{CHEM}%
  \togglefalse{ELEC}\toggletrue{ENG}\togglefalse{SCI}%
}%
\DeclareOption{sci}{%
  \togglefalse{ARTS}\togglefalse{BIZ}\togglefalse{CHEM}%
  \togglefalse{ELEC}\togglefalse{ENG}\toggletrue{SCI}%
}%
%% Font encoding support for iso-latin-1, utf-8 and ansinew (MiKTeX):
%% LC 26.9.2013
\newtoggle{UTF}%
\DeclareOption{latin1}{%
  \PassOptionsToPackage{latin1}{inputenc}%
}%
\DeclareOption{utf8}{%
  \PassOptionsToPackage{utf8}{inputenc}%
  \toggletrue{UTF}%
}%
\DeclareOption{ansinew}{%
  \PassOptionsToPackage{ansinew}{inputenc}%
}%
%
\newtoggle{PDFA}%
\newtoggle{writexmpdatafile}
%
% PDF/A support extended from a-1b to a-2b. LC 17.4.2018
\DeclareOption{a-1b}{\toggletrue{PDFA}%
  \toggletrue{writexmpdatafile}%
  \PassOptionsToPackage{a-1b}{pdfx}%
}%
%
\DeclareOption{a-2b}{%
  \toggletrue{PDFA}\toggletrue{writexmpdatafile}%
  \PassOptionsToPackage{a-2b}{pdfx}%
}%
% Support for PDF/A-3b removed to avoid problem for users. LC 31.8.2018
%\DeclareOption{a-3b}{%
%  \toggletrue{PDFA}\toggletrue{writexmpdatafile}%
%  \PassOptionsToPackage{a-3b}{pdfx}%
%}%
%
\newtoggle{ONLINE}%
\DeclareOption{online}{%
  \toggletrue{ONLINE}%
}%
\DeclareOption*{%
  \PassOptionsToClass{\CurrentOption}{article}%
}%
\ExecuteOptions{english, utf8, online, a4paper}% Default package options
\ProcessOptions\relax%
%
\LoadClass{article}%
%
\iftoggle{PDFA}{%
  \RequirePackage[latxmp]{pdfx}[2017/05/18]% Produce PDF/A 1b, 2b and 3b compliant pdf
  \RequirePackage{newfile}% Provides an easy interface for writing data into new
						  % file (used to write metadata into file .xmpdata)
  \iftoggle{UTF}{\hypersetup{pdfencoding=unicode}}{}%
  \hypersetup{pdfpagemode=UseNone, bookmarksnumbered=true}%
  \iftoggle{ONLINE}{%
  	\hypersetup{colorlinks=true, urlcolor=blue, linkcolor=blue, citecolor=blue}%
  }{}%
}{%
  \AtEndPreamble{%
	\RequirePackage{hyperxmp}%
	\RequirePackage{hyperref}%
  }%
}%
%
\AtEndPreamble{% Set up the hyperref options
  \iftoggle{UTF}{%
  	\hypersetup{pdfencoding=unicode}%
  }{}%
  \hypersetup{pdfpagemode=UseNone, bookmarksnumbered=true}%
  \iftoggle{ONLINE}{%
  	\hypersetup{colorlinks=true, urlcolor=blue, linkcolor=blue, citecolor=blue}%
  }{%
	\hypersetup{colorlinks=false}
  }%
}%
%
\RequirePackage[TS1,T1]{fontenc}%
\RequirePackage{inputenc}% For scands in both Finnish and English input docs
\RequirePackage[english, finnish, swedish]{babel}%
\RequirePackage{lmodern}%
\RequirePackage[kerning, spacing, babel]{microtype}%
%
\RequirePackage{calc}%
\RequirePackage{chngcntr}% Allows counter dependencies to be changed for appendix
%
%
%% Set up the logo
%% Note that BIZ and ARTS still (2012-08-17) use old logos as the
%% aaltologo-package is not up-to-date. Hence there are calls to ECON and TAIK
%% pic routines. 
\iftoggle{ARTS}{% 
  \RequirePackage[TAIK, Uncoated]{aaltologo}%
}{%
\iftoggle{BIZ}{%
  \RequirePackage[ECON, Uncoated]{aaltologo}%
}{%
\iftoggle{CHEM}{%
  \RequirePackage[CHEM, Uncoated]{aaltologo}%
}{%
\iftoggle{ELEC}{%
   \RequirePackage[ELEC, Uncoated]{aaltologo}%
}{%
\iftoggle{ENG}{%
  \RequirePackage[ENG, Uncoated]{aaltologo}%
}{%
\iftoggle{SCI}{%
  \RequirePackage[SCI, Uncoated]{aaltologo}%
}{%
\RequirePackage[Uncoated]{aaltologo}%
}}}}}}%
%
%%%%%%% Set up the page layout %%%%%%%
%% First the horizontal dimensions %%
\iftoggle{ONLINE}{%
  \setlength{\oddsidemargin}{4.6mm}% 1" + 4.6mm = 3cm
  \setlength{\evensidemargin}{4.6mm}%
}{%
  \setlength{\oddsidemargin}{9.6mm}% 3.5cm
  \setlength{\evensidemargin}{-0.4mm}% 1" - 0.4mm = 2.5cm
}%
\setlength{\textwidth}{15cm}%
%% Then the vertical dimensions
\setlength{\headsep}{12mm}%
\setlength{\headheight}{1em}%
\setlength{\topmargin}{-5mm}%
%
\setlength{\textheight}{23cm}%
%
%%%%%%% Float control %%%%%%%
\renewcommand{\floatpagefraction}{0.1}%
\renewcommand{\textfraction}{0.1}%
\renewcommand{\topfraction}{0.9}%
\renewcommand{\bottomfraction}{0.9}%
%
%%%%%% Now come all the macros used along with all the required counter, etc.
%
\newcommand*{\AbstractLang}{\MainLang}%
%
%%%%%%% Some shorthand %%%%%%%
\newcommand*{\IfEng}[3]{\ifthenelse{\equal{#1}{english}}{#2}{#3}}%
\newcommand*{\IfFin}[3]{\ifthenelse{\equal{#1}{finnish}}{#2}{#3}}%
\newcommand*{\IfSwe}[3]{\ifthenelse{\equal{#1}{swedish}}{#2}{#3}}%
%
%%%%%%% Storage of number of pages and number of abstracts %%%%%%%
\newcounter{NAllPages}%
\newcounter{NMainTxtPages}%
\newcounter{NAppdxPages}%
% write in file .aux
\newcommand*{\stor@maintxtpag@number}{%
  \immediate\write\@auxout{\string\setcounter{NMainTxtPages}{\thepage-1}}%
}%
\AtEndDocument{%
  \immediate\write\@auxout{\string\setcounter{NAllPages}{\thepage}}%
}%
%
%%%%%%% Default fonts: Latin Modern (LC 24.9.2015)
% This is required to overwite any fonts set elsewhere. The package 
% aaltologo.sty sets \rmdefault to newcent, \ttdefault to txtt.
\renewcommand{\rmdefault}{lmr}%
\renewcommand{\ttdefault}{lmtt}%
\renewcommand{\sfdefault}{lmss}%
%
%%%%%%
%%%%%% Language-specific definitions
%%%%%% First create the necessary variables.
%%%%%%
\newcommand*{\univ@rsity}{}%
\newcommand*{\sch@@l}{}%
\newcommand*{\@ddress}{}%
\newcommand*{\urln@me}{www.aalto.fi}%
\newcommand*{\thesissupervisorname}{}%
\newcommand*{\thesissuperadvisorname}{}%
\newcommand*{\thesisadvisorname}{}%
\newcommand*{\thesisadvisorsname}{}%
\newcommand{\covertext}{}%
\newcommand*{\authorname}{}%
\newcommand*{\thesisname}{}%
\newcommand*{\keywordsname}{}%
\newcommand*{\datename}{}%
\newcommand*{\langname}{}%
\newcommand*{\nofpgsname}{}%
\newcommand*{\supervisorname}{}%
\newcommand*{\advisorname}{}%
\newcommand*{\degprogramname}{}%
\newcommand*{\departmentname}{}%
\newcommand*{\degreeordep@rtmentname}{}%
\newcommand*{\degreeordep@rtment}{}%
%\newcommand*{\professorshipname}{}%
\newcommand*{\majorname}{}%
\newcommand*{\codename}{}%
\newcommand*{\tocabstractname}{}%
%%
\newcommand*{\absheadname}{}%
\newcommand*{\tocabslang}{}%
%
%%%%%% Set up those values that can be set up automatically: university,
% school name, logo, etc.
%% Some can also be set explicitly by the user via macros like \university
%% and \school defined below after the language-specific stuff.
%%
\newcommand{\setthesismainlangnames}{%
  \IfEng{\MainLang}{%
    %% Cover page stuff in English
    \iftoggle{ARTS}{%
	  \renewcommand*{\sch@@l}{School of Arts, Design and Architecture}%
    }{%
    \iftoggle{BIZ}{%
      \renewcommand*{\sch@@l}{School of Business}%
    }{%
    \iftoggle{CHEM}{%
	  \renewcommand*{\sch@@l}{School of Chemical Technology}%
    }{%
    \iftoggle{ELEC}{%
	  \renewcommand*{\sch@@l}{School of Electrical Engineering}%
    }{%
    \iftoggle{ENG}{%
	  \renewcommand*{\sch@@l}{School of Engineering}%
    }{%
    \iftoggle{SCI}{%
	  \renewcommand*{\sch@@l}{School of Science}%
    }{%
    \ifdefstring{\sch@@l}{}{%
	  \PackageError{aaltothesis}{%
		Your school isn't specified%
	  }{%
		Specify arts, biz, chem, elec eng or sci in \protect\documentclass{} %
		or specify it using \protect\school{}.%
	  }%
	}{}%
	}}}}}}%
    \renewcommand*{\thesissupervisorname}{Supervisor}%
    \renewcommand*{\thesissuperadvisorname}{Supervisor and advisor}%
    \renewcommand*{\thesisadvisorname}{Advisor}%
    \renewcommand*{\thesisadvisorsname}{Advisors}%
    \renewcommand{\covertext}{%
      \iftoggle{BSc}{%
      	Bachelor's thesis%
      }{%
      \iftoggle{MSc}{%
        Thesis submitted for examination for the degree of Master of Science in %
        Technology.%
      }{%
      \iftoggle{Lic}{%
        Thesis submitted for examination for the degree of Licentiate of Science %
        in Technology.%
      }{}}}%
    }%
    \renewcommand*{\tocabstractname}{Abstract}%
  }{% end if English
  \IfFin{\MainLang}{%
    %% Cover page stuff in Finnish
    \iftoggle{ARTS}{%
	  \renewcommand*{\sch@@l}{Taiteiden ja suunnittelun korkeakoulu}%
    }{%
    \iftoggle{BIZ}{%
	  \renewcommand*{\sch@@l}{Kauppakorkeakoulu}%
    }{%
    \iftoggle{CHEM}{%
	  \renewcommand*{\sch@@l}{Kemian tekniikan korkeakoulu}%
    }{%
    \iftoggle{ELEC}{%
	  \renewcommand*{\sch@@l}{S\"ahk\"otekniikan korkeakoulu}%
    }{%
    \iftoggle{ENG}{%
	  \renewcommand*{\sch@@l}{Insin\"o\"oritieteiden korkeakoulu}%
    }{%
    \iftoggle{SCI}{%
	  \renewcommand*{\sch@@l}{Perustieteiden korkeakoulu}%
    }{%
    \ifdefstring{\sch@@l}{}{%
      \PackageError{aaltothesis}{%
      	Your school isn't specified%
      }{%
        Specify arts, biz, chem, elec eng or sci in \protect\documentclass{} %
        or specify it using \protect\school{}.%
      }%
    }{}%
	}}}}}}%
    \renewcommand*{\thesissupervisorname}{%
      \iftoggle{BSc}{Vastuuopettaja}{Ty\"on valvoja}%
    }%
    \renewcommand*{\thesissuperadvisorname}{%
      \iftoggle{BSc}{%
	    Vastuuopettaja ja ohjaaja%
	  }{%
	    Ty\"on valvoja ja ohjaaja%
	  }%
    }%
    \renewcommand*{\thesisadvisorname}{Ty\"on ohjaaja}%
    \renewcommand*{\thesisadvisorsname}{Ty\"on ohjaajat}%
    \renewcommand{\covertext}{%
      \iftoggle{BSc}{%
		Kandidaatinty\"o%
      }{%
      \iftoggle{MSc}{%
        Diplomity\"o, joka on j\"atetty opinn\"aytteen\"a tarkastettavaksi %
        diplomi-insin\"o\"orin tutkintoa varten Espoossa \D@te{}.%
      }{%
      \iftoggle{Lic}{%
        Lisensiaatintutkimus, joka on j\"atetty opinn\"aytteen\"a tarkastettavaksi %
        tekniikan lisensiaatin tutkintoa varten Espoossa \D@te{}.
      }{}}}%
    }%
    \renewcommand*{\tocabstractname}{Tiivistelm\"a}%
    \renewcommand{\contentsname}{Sis\"allysluettelo}%
  }{% end if Finnish
  \IfSwe{\MainLang}{%
    %% Cover page stuff in Swedish
    \iftoggle{ARTS}{%
	  \renewcommand*{\sch@@l}{H\"ogskolan f\"or konst, design och arkitektur}%
    }{%
    \iftoggle{BIZ}{%
	  \renewcommand*{\sch@@l}{Handelsh\"ogskolan}%
    }{%
    \iftoggle{CHEM}{%
	  \renewcommand*{\sch@@l}{H\"ogskolan f\"or kemiteknik}%
    }{%
    \iftoggle{ELEC}{%
	  \renewcommand*{\sch@@l}{H\"ogskolan f\"or elektroteknik}%
    }{%
    \iftoggle{ENG}{%
	  \renewcommand*{\sch@@l}{H\"ogskolan f\"or ingenj\"orsvetenskaper}%
    }{%
    \iftoggle{SCI}{%
	  \renewcommand*{\sch@@l}{H\"ogskolan f\"or teknikvetenskaper}%
    }{%
    \ifdefstring{\sch@@l}{}{%
	  \PackageError{aaltothesis}{%
		Your school isn't specified%
	  }{%
		Specify arts, biz, chem, elec eng or sci in \protect\documentclass{} %
		or specify it using \protect\school{}.%
	  }%
	}{}%
	}}}}}}%
    \renewcommand*{\thesissupervisorname}{%
      \iftoggle{BSc}{%
      	Ansvarsl\"arare%
      }{%
        \"Overvakare
      }%
    }%
    \renewcommand*{\thesissuperadvisorname}{%
      \iftoggle{BSc}{%
       Ansvarsl\"arare och handledare%
      }{%
        Arbetets \"overvakare och handledare%
      }%
    }%
    \renewcommand*{\thesisadvisorname}{Handledare}%
    \renewcommand*{\thesisadvisorsname}{Handledare}%
    \renewcommand{\covertext}{%
      \iftoggle{BSc}{%
      	Kandidatexamen%
      }{%
      \iftoggle{MSc}{%
        Diplomarbete, som inl\"amnats f\"or granskning som l\"ardomsprov
        f\"or avl\"aggande av diplomingenj\"orsexamen i Esbo den \D@te{}.%
      }{%
      \iftoggle{Lic}{%
        Licentiatavhandling, som inl\"amnats f\"or granskning som l\"ardomsprov
        f\"or avl\"aggande av licentiatexamen i Esbo den \D@te{}.%
      }{}}}%
    }%
    \renewcommand*{\tocabstractname}{Sammandrag}%
    \renewcommand{\contentsname}{Inneh\aa{}llsf\"orteckning}%
  }{% end if Swedish
  }}}% Close the else branch of all three languages
}%
%
\newcommand*{\setabstractlangnames}{%
% Define the the headers, the names of fields in the language of the abstract,
% and set up some lengths used in the abstract.
 \IfEng{\AbstractLang}{%
   %% Abstract header and field names in English
   \renewcommand*{\univ@rsity}{Aalto University}%
   \renewcommand*{\@ddress}{\univ@rsity, P.O. BOX 11000, 00076 AALTO}
   \renewcommand*{\authorname}{Author}%
   \renewcommand*{\thesisname}{Title}%
   \renewcommand*{\abstractname}{Abstract}%
   \renewcommand*{\keywordsname}{Keywords}%
   \renewcommand*{\datename}{Date}%
   \renewcommand*{\langname}{Language}%
   \renewcommand*{\nofpgsname}{Number of pages}%
   \renewcommand*{\supervisorname}{%
     \ifnumequal{\value{advisors}}{0}{%
       \iftoggle{BSc}{%
		 Teacher in charge and advisor%
	   }{%
		 Supervisor and advisor%
	   }%
     }{%
       \iftoggle{BSc}{%
		 Teacher in charge%
	   }{%
		 Supervisor%
	   }%
    }%
   }%
   \renewcommand*{\advisorname}{%
     \ifnumequal{\value{advisors}}{0}{}{%
       \ifnumequal{\value{advisors}}{1}{Advisor}{%
         \ifnumgreater{\value{advisors}}{1}{Advisors}{%
     }}}%%%
   }%
   %
   \renewcommand*{\degprogramname}{Degree programme}%
   \renewcommand*{\departmentname}{Department}%
   %
   \renewcommand*{\majorname}{Major}%
   \renewcommand*{\codename}{Code of major}%
   \iftoggle{BSc}{%
     \renewcommand*{\absheadname}{Abstract of the bachelor's thesis}%
   }{%
   \iftoggle{MSc}{%
     \renewcommand*{\absheadname}{Abstract of the master's thesis}%
   }{%
   \iftoggle{Lic}{%
     \renewcommand*{\absheadname}{Abstract of the licentiate's thesis}%
   }{}}}%
 }{% end if English
 \IfFin{\AbstractLang}{%
   %% Abstract header and field names in Finnish
   \renewcommand*{\univ@rsity}{Aalto-yliopisto}%
   \renewcommand*{\@ddress}{\univ@rsity, PL 11000, 00076 AALTO}
   \renewcommand*{\authorname}{Tekij\"a}%
   \renewcommand*{\thesisname}{Ty\"on nimi}%
   \renewcommand*{\abstractname}{Tiivistelm\"a}%
   \renewcommand*{\keywordsname}{Avainsanat}%
   \renewcommand*{\datename}{P\"aiv\"am\"a\"ar\"a}%
   \renewcommand*{\langname}{Kieli}%
   \renewcommand*{\nofpgsname}{Sivum\"a\"ar\"a}%
   \renewcommand*{\supervisorname}{%
     \ifnumequal{\value{advisors}}{0}{%
       \iftoggle{BSc}{%
         Vastuuopettaja ja ohjaaja%
       }{%
         Ty\"on valvoja ja ohjaaja%
       }%
     }{%
       \iftoggle{BSc}{%
       	 Vastuuopettaja%
        }{%
          Ty\"on valvoja%
        }%
     }%
   }%
   \renewcommand*{\advisorname}{%
     \ifnumequal{\value{advisors}}{0}{}{%
       \ifnumequal{\value{advisors}}{1}{Ty\"on ohjaaja}{%
         \ifnumgreater{\value{advisors}}{1}{Ty\"on ohjaajat}{%
     }}}%%%
   }%
   \renewcommand*{\degprogramname}{Koulutusohjelma}%
   \renewcommand*{\departmentname}{Laitos}%
   \renewcommand*{\majorname}{P\"a\"aaine}%
   \renewcommand*{\codename}{P\"a\"aaineen koodi}%
   \iftoggle{BSc}{%
     \renewcommand*{\absheadname}{Tekniikan kandidaatinty\"on tiivistelm\"a}%
   }{%
   \iftoggle{MSc}{%
     \renewcommand*{\absheadname}{Diplomity\"on tiivistelm\"a}%
   }{%
   \iftoggle{Lic}{%
     \renewcommand*{\absheadname}{Lisensiaatintutkimuksen tiivistelm\"a}%
   }{}}}%
 }{% end if Finnish
 \IfSwe{\AbstractLang}{%
   %% Abstract header and field names in Swedish
   \renewcommand*{\univ@rsity}{Aalto-universitetet}%
   \renewcommand*{\@ddress}{\univ@rsity, PB 11000, 00076 AALTO}%
   \renewcommand*{\authorname}{F\"orfattare}%
   \renewcommand*{\thesisname}{Titel}%
   \renewcommand*{\abstractname}{Sammandrag}%
   \renewcommand*{\keywordsname}{Nyckelord}%
   \renewcommand*{\datename}{Datum}%
   \renewcommand*{\langname}{Spr\r{a}k}%
   \renewcommand*{\nofpgsname}{Sidantal}%
   \renewcommand*{\supervisorname}{%
     \ifnumequal{\value{advisors}}{0}{%
       \iftoggle{BSc}{%
         Ansvarsl\"arare%
       }{%
         \"Overvakare och handledare%
       }%
     }{%
       \iftoggle{BSc}{%
       	 Ansvarsl\"arare%
       }{%
         \"Overvakare%
       }%
     }%
   }%
   \renewcommand*{\advisorname}{%
     \ifnumequal{\value{advisors}}{0}{}{%
       \ifnumequal{\value{advisors}}{1}{Handledare}{%
         \ifnumgreater{\value{advisors}}{1}{Handledare}{%
     }}}%%%
   }%
   \renewcommand*{\degprogramname}{Utbildningsprogram}%
   \renewcommand*{\departmentname}{Institution}%
   \renewcommand*{\majorname}{Huvud\"amne}%
   \renewcommand*{\codename}{Huvud\"amnets kod}%
   \iftoggle{BSc}{%
     \renewcommand*{\absheadname}{Sammandrag av kandidatarbetet}%
   }{%
   \iftoggle{MSc}{%
     \renewcommand*{\absheadname}{Sammandrag av diplomarbetet}%
   }{%
   \iftoggle{Lic}{%
     \renewcommand*{\absheadname}{Sammandrag av licentiatavhandlingen}{}%
   }{}}}%
 }{% end if Swedish
 }}}% Close the else branch of all three languages
% The requirements have changed here again. The masters thesis should no longer
% have 'Department' and 'Professorship', but should be like for the bachelors
% thesis. I haven't removed this now unnecessary \iftoggle statement in case
% this requirement changes again.
 \iftoggle{BSc}{%
   \renewcommand*{\degreeordep@rtmentname}{\degprogramname}%
   \renewcommand*{\degreeordep@rtment}{\d@gpr@g}% \d@gpr@g is set in \degreeprogram
 }{%
%   \renewcommand*{\degreeordep@rtmentname}{\departmentname}%
%   \renewcommand*{\degreeordep@rtment}{\dep@rtment}% \dep@rtment is set in \department
   \renewcommand*{\degreeordep@rtmentname}{\degprogramname}%
   \renewcommand*{\degreeordep@rtment}{\d@gpr@g}% \d@gpr@g is set in \degreeprogram
 }%
}%
%
\newcommand*{\L@ng}[1]{% internal macro:
% Determine the language of the thesis (\MainLang) provided by the user in the
% \documentclass options and write it into the abstract in the appropriate field
% field in the language of the current abstract. The abstract language is
% the parameter of this macro, and it is provided by the user in the abstract
% environment.
  \ifstrequal{#1}{english}{%
    \IfEng{\MainLang}{English}{%
    \IfFin{\MainLang}{Finnish}{%
    \IfSwe{\MainLang}{Swedish}{%
    }}}%
  }{%
  \ifstrequal{#1}{finnish}{%
    \IfEng{\MainLang}{Englanti}{%
    \IfFin{\MainLang}{Suomi}{%
    \IfSwe{\MainLang}{Ruotsi}{%
    }}}%
  }{%
  \ifstrequal{#1}{swedish}{%
    \IfEng{\MainLang}{Engelska}{%
    \IfFin{\MainLang}{Finska}{%
    \IfSwe{\MainLang}{Svenska}{%
    }}}%
  }{%
  }}}%
}%
%
\newcommand{\settocabslang}{% internal macro:
% Sets value of \tocabslang, the extra text that goes into the toc when there
% are more abstracts than one. The value of \tocabslang has to be set explicitly
% prior to use, since it is used in \addcontentsline in the abstractpage
% environment, and so it may not contain any \if* commands. Thus making
% \tocabslang decide with an \if* results in an error since it is part of the
% argument of \addcontentsline.
  \ifdefstrequal{\MainLang}{\AbstractLang}{%
    \renewcommand*{\tocabslang}{}%
  }{%
    \IfEng{\MainLang}{%
      \IfFin{\AbstractLang}{%
      	\renewcommand*{\tocabslang}{(in Finnish)}%
      }{%
      \IfSwe{\AbstractLang}{%
      	\renewcommand*{\tocabslang}{(in Swedish)}%
      }{%
        \PackageError{aaltothesis}{%
          Gosh! You are in dark waters! \protect\settocabslang (1)%
		}{%
		  Perhaps you need to correct all previous errors.%
	    }%
      }}%
    }{%
    \IfFin{\MainLang}{%
      \IfEng{\AbstractLang}{%
      	\renewcommand*{\tocabslang}{(englanniksi)}%
      }{%
      \IfSwe{\AbstractLang}{%
      	\renewcommand*{\tocabslang}{(ruotsiksi)}%
      }{%
        \PackageError{aaltothesis}{%
          Gosh! You are in dark waters! \protect\settocabslang (2)%
        }{%
		  You may need to correct all previous errors. May the force be with you.%
	    }%
      }}%
    }{%
    \IfSwe{\MainLang}{%
      \IfEng{\AbstractLang}{%
      	\renewcommand*{\tocabslang}{(p\aa{} engelska)}%
      }{%
      \IfFin{\AbstractLang}{%
      	\renewcommand*{\tocabslang}{(p\aa{} finska)}%
      }{%
		\PackageError{aaltothesis}{%
		  Gosh! You are in dark waters! \protect\settocabslang (3)%
	    }{%
		  You may need to correct all previous errors. May the force be with you.%
	    }%
      }}%
    }{%
      \PackageError{aaltothesis}{%
		Gosh! You should not have got this far. \protect\settocabslang (4)%
      }{%
		You may need to correct all previous errors. May the force be with you.%
      }%
    }}}%
  }%
}%
%
%% This ends the definitions of language-specific variables
%
\AtEndPreamble{%
  \ifboolexpr{ togl {BSc} or togl {MSc} or togl {Lic} }{%
	\setthesismainlangnames%
	\setabstractlangnames%
  }{%
	\PackageError{aaltothesis}{%
	  \protect\univdegree{} is not specified%
	}{%
	  Specify BSc, MSc or Lic in \protect\univdegree{}.%
	}%
  }%
}%
%
%%%%%% Interface commands: parameters specified by user (+ necessary variables).
%% Some have been automatically set up in the language-specific part but can
%% be overrun by the user via these macros.
%% First the straightforward ones:
%
\newcommand*{\ThesisAuthor}{}% variable
\newcommand*{\thesisauthor}[1]{\renewcommand*{\ThesisAuthor}{#1}}% User command
%
\newcommand{\th@sistitl@}{}% variable
\newcommand*{\pdfth@sistitl@}{}% variable
\newcommand{\thesistitle}[2][]{% User command
  \renewcommand{\th@sistitl@}{#2}%
  \ifstrempty{#1}{%
	\renewcommand*{\pdfth@sistitl@}{#2}%
  }{%
	\renewcommand*{\pdfth@sistitl@}{#1}%
  }%
}%
%
\newcommand*{\D@te}{}% variable
\renewcommand*{\date}[1]{\renewcommand*{\D@te}{#1}}% User command
%
\newcommand*{\pl@ce}{}% variable
\newcommand*{\place}[1]{\renewcommand*{\pl@ce}{#1}}% User command
%
\newcommand*{\dateandplace}{\pl@ce{} \D@te{}}% Internal macro
%
%\newcommand*{\university}[1]{\renewcommand*{\univ@rsity}{#1}}% User command
%
\newcommand*{\school}[1]{\renewcommand*{\sch@@l}{#1}}% User command
%
\newcommand*{\d@gpr@g}{}% variable
\newcommand*{\degreeprogram}[1]{\renewcommand*{\d@gpr@g}{#1}}% User command
%
\newcommand*{\sup@rvis@r}{}% variable
\newcommand*{\supervisor}[1]{\renewcommand*{\sup@rvis@r}{#1}}% User command
%
% user interface for logo
\newcommand*{\@@ltologo}{}% variable
\newcommand*{\@@ltoabslogo}{}% variable
\newcommand{\uselogo}[2]{%
  \renewcommand*{\@@ltologo}{\AaltoLogoSmall{1}{#2}{#1}}% for the cover page
  \renewcommand*{\@@ltoabslogo}{\AaltoLogoSmall{0.8}{#2}{#1}}% for the abstract page
}%
%
\newcommand*{\m@jor}{}% variable
\newcommand*{\major}[1]{\renewcommand*{\m@jor}{#1}}%
% User commands for M.Sc. and Lic. only
%\newcommand*{\professorship}[1]{\renewcommand*{\m@jor}{#1}}%
%
\newcommand*{\Cod@}{}% variable
\newcommand*{\code}[1]{\renewcommand{\Cod@}{#1}}% User command
%
\newcommand*{\dep@rtment}{}% variable
\newcommand*{\department}[1]{\renewcommand*{\dep@rtment}{#1}}% User command
%
% Then come the more complex macros:
% First some boolean flags
\newtoggle{BSc}% default is false
\newtoggle{MSc}%
\newtoggle{Lic}%
\newcommand*{\univdegree}[1]{% User command
  \ifstrequal{#1}{BSc}{%
    \toggletrue{BSc}\togglefalse{MSc}\togglefalse{Lic}%
  }{%
  \ifstrequal{#1}{MSc}{%
    \togglefalse{BSc}\toggletrue{MSc}\togglefalse{Lic}%
  }{%
  \ifstrequal{#1}{Lic}{%
    \togglefalse{BSc}\togglefalse{MSc}\toggletrue{Lic}%
  }{%
    \PackageError{aaltothesis}{%
	  Only BSc, MSc or Lic is allowed as parameter for degree%
	}{%
	  Use BSc, MSc or Lic.%
    }%
  }}}%
}%
%
\newcounter{advisors}%
\newcommand*{\advis@rA}{}%
\newcommand*{\advis@rB}{}%
\newcommand*{\advisor}[1]{% User command
  \stepcounter{advisors}%
  \ifnumequal{\value{advisors}}{1}{%
    \renewcommand*{\advis@rA}{#1}%
  }{%
  \ifnumequal{\value{advisors}}{2}{%
    \renewcommand*{\advis@rB}{#1}%
  }{%
    \PackageError{aaltothesis}{%
      Only 2 advisors are allowed%
	}{%
	  Give at most 2 advisors or hack the code.
    }%
  }}%
}%
\newcommand*{\Advis@r}[1]{% internal macro
  \ifstrequal{#1}{1}{% One advisor
    \advis@rA{}%
  }{%
  \ifstrequal{#1}{2}{% Two advisors
    \advis@rB{}%
  }{%
    \PackageError{aaltothesis}{%
      Only 2 advisors are allowed%
    }{%
      Give at most 2 advisors or hack the code.%
    }%
  }}%
}%
%
%%
%% Two cover-related internal macros
%%
\newcommand*{\cov@rsup@rvisor}{%
%  \ifdefstrequal{\sup@rvis@r}{\advis@rA}{% same supervisor & advisor
%    \global\defcounter{advisors}{0}%
%    \thesissuperadvisorname%
%  }{%
%    \thesissupervisorname%
%  }%
  \ifnumequal{\value{advisors}}{0}{%
  	\thesissuperadvisorname%
  }{%
    \thesissupervisorname%
  }%
}%
%
\newcommand*{\cov@radvisor}{%
  \ifnumequal{\value{advisors}}{0}{%
  }{%
  \ifnumequal{\value{advisors}}{1}{%
	\thesisadvisorname%
  }{%
  \ifnumgreater{\value{advisors}}{1}{%
	\thesisadvisorsname%
  }{%
	\PackageError{aaltothesis}{%
	  Uh oh! You should never have got here (\protect\cov@radvisor).%
    }{%
      You're in dark waters now...
    }%
  }}}%
}%
%
%%
%% Interface for copyright text
%%
\newcommand{\CopyrightText}{}% variable; used for printing on page
\newcommand{\MetaCopyrightText}{}%
\newcommand{\copyrighttext}[2]{%
  \renewcommand{\MetaCopyrightText}{#1}%
  \renewcommand{\CopyrightText}{#2}%
}%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Do the cover page layout %%%%%%%%%%%%%%%%
\pagenumbering{arabic}%
\newlength{\coverpageindent}%
\setlength{\coverpageindent}{10mm}%
\newlength{\coverindent}%
\setlength{\coverindent}{48mm}%
\newcommand{\mytextsf}[1]{%
  {\usefont{T1}{phv}{b}{n}\fontsize{11}{13.2}\selectfont{#1}}%
}%
\newcommand{\makecoverpage}{%
  \thispagestyle{empty}%
  \vspace*{3.5mm}%
  \noindent% Position the title and author
  % The title and author, each in their own box, are placed in an enclosing
  % box with its height limited so as to 
  % 1. fix the position of the text that follows on the page and
  % 2. to prevent unnecessarily long titles.
  % The vertical space for the title is 120pt, followed by a 6-mm blank
  % space, and the author's name is typeset in a 12-pt high font.
  \hspace*{\coverpageindent}%
  \parbox[t][132pt+6mm]{0.75\textwidth-\coverpageindent}{%
    \noindent% First position the title
    \parbox[t]{0.75\textwidth-\coverpageindent}{\raggedright%
      \usefont{T1}{phv}{b}{n}\fontsize{18}{21}\selectfont{\th@sistitl@}}\par%
    \vspace{8mm}%
    \noindent% followed by the author
    \parbox{0.75\textwidth-\coverpageindent}{\large\textbf{\ThesisAuthor}}\par%
  }\par%
  \vspace{26mm}%
  \noindent%
  \hspace*{\coverpageindent}%
  \parbox{0.75\textwidth-\coverpageindent}{\raggedright\small%
    {\usefont{T1}{phv}{b}{n}\fontsize{11}{13.2}\selectfont{\sch@@l}}\\[1em]% Position the school
    \covertext% Position the degree-work text
  }\par%
  \vspace{1ex}%
  \noindent%
  \hspace*{\coverpageindent}%
  \parbox{0.80\textwidth-\coverpageindent}{%
    \raggedright%
    \small%
    % Place the date and place; only for the BSc text and text in English
    \ifboolexpr{togl {BSc} or test {\ifdefstring{\MainLang}{english}}}{%
      \dateandplace\par%
    }{}%
    \vspace{23mm}%
    \noindent%
    \mytextsf{\cov@rsup@rvisor}\par%
    \vspace{1em}%
    \hspace{\coverindent}%
    \parbox{\linewidth-\coverindent}{\raggedright\sup@rvis@r}\par%
    \ifnumgreater{\value{advisors}}{0}{%
      \vspace{2em}%
      \noindent%
      \mytextsf{\cov@radvisor{}}\par%
      \vspace{1em}%
      \hspace{\coverindent}%
      \parbox{\linewidth-\coverindent}{\raggedright\Advis@r{1}}%
    }{}%
    \ifnumgreater{\value{advisors}}{1}{%
      \par%
      \vspace{4em}%
      \hspace{\coverindent}\parbox{\linewidth-\coverindent}{\raggedright\Advis@r{2}}%
    }{}%
  }%
  \vfill%
  \noindent%
  \hspace{\coverpageindent}%
  \@@ltologo%
  \clearpage%
  \pagestyle{myheadings}%
}%
%
%%%%%%%%%%%%%% End of the cover page layout %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% The copyright page %%%%%%%%%%%%%%
\newcommand{\makecopyrightpage}{%
  \thispagestyle{empty}%
  \vspace*{3.5mm}%
  \CopyrightText\par%
  \clearpage%
  \pagestyle{myheadings}%
}%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Abstract stuff begins here %%%%%%%%%%%%%%%
%% First some lengths and internal macros
\newlength{\absheadlen}% Used to set width of 2nd column of the abstract header,
% this width is set above according to the respective text width
\newlength{\abstractwidth}%
\AtBeginDocument{\setlength{\abstractwidth}{\textwidth}}%
\newlength{\rowsep}%
\setlength{\rowsep}{-1.3\columnsep}%
\newlength{\keywdparheight}%
\setlength{\keywdparheight}{3\baselineskip}%
\newlength{\auxwidth}%
\newlength{\fnamelenA}%
\newlength{\fnamelenB}%
\newlength{\ftextlen}%
%
% the line across the abstract page (with space around it)
\newcommand*{\@bsbre@k}{%
  \vspace{2.5pt}\hrule\vspace{3.5pt}%
}%
% two boxes, one for the field name and the other for the content
\newcommand{\@bsfield}[3][]{%
  \settowidth{\auxwidth}{\mytextsf{#2}}%
  \noindent%
  \parbox[t][#1]{\auxwidth}{\mytextsf{#2}}%
  \hspace{1ex}%
  \parbox[t][#1]{\abstractwidth-\auxwidth-1ex}{\raggedright#3}%
  \@bsbre@k%
}%
%
\newcommand*{\shownumberofpages}{%
  \ifthenelse{\value{NMainTxtPages}=0}{\arabic{NAllPages}}%
 	{\arabic{NMainTxtPages}+\arabic{NAppdxPages}}%
}%
%
\newcommand{\k@ywords}{}%
\newcommand{\keywords}[1]{\renewcommand*{\k@ywords}{#1}}%
%
%% The macro to contain the abstract text so that it can be added to the
%% metadata in the xmpdata file
\newcommand{\abstracttext}{}%
\newcommand{\thesisabstract}[1]{\renewcommand{\abstracttext}{#1}}%
\newcommand*{\spc}{\noexpand\sep}%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% The abstract page layout %%%%%%%%%%%%%%%%%%%%%%%%%%%
\newenvironment{abstractpage}[1][\MainLang]{%
  \ifstrequal{#1}{english}{%
  	\renewcommand*{\AbstractLang}{english}%
  }{%
  \ifstrequal{#1}{finnish}{%
  	\renewcommand*{\AbstractLang}{finnish}%
  }{%
  \ifstrequal{#1}{swedish}{%
  	\renewcommand*{\AbstractLang}{swedish}%
  }{%
    \PackageError{aaltothesis}{%
      Only english, finnish or swedish is allowed as optional parameter%
    }{%
      Use english, finnish or swedish.%
    }%
  }}}%
  \setabstractlangnames%
  \settowidth{\absheadlen}{\@ddress}%
  \settowidth{\fnamelenA}{\mytextsf{\majorname}}%
  \settowidth{\fnamelenB}{\mytextsf{\codename}}%
  \settowidth{\ftextlen}{\Cod@}%
  \phantomsection%
  \settocabslang%
  \addcontentsline{toc}{section}{\tocabstractname{} \tocabslang}%
  \renewcommand*{\spc}{, }
  \setcounter{NAppdxPages}{\value{NAllPages}-\value{NMainTxtPages}}% no. of pages in appendix
  \thispagestyle{empty}%
  \enlargethispage{5mm}%
  \vspace*{-4em}%
  \noindent%
% First the header of the abstract page
  \begin{tabular}{@{}p{\abstractwidth-\absheadlen}@{}p{\absheadlen}@{}}%
	\mbox{}\par\vspace*{-0.75em}\@@ltoabslogo% this is the logo
    &% here is the address, url and degree name
	\small\raggedleft%
  	\mytextsf{\@ddress}\par\vspace{-0.5ex}%
  	\mytextsf{\urln@me}\par\vspace{-0.4ex}%
  	\mytextsf{\absheadname}%
  \end{tabular}%
  \par\vspace{3em}%
%% Then the abstract form
  \noindent%
  \hrule\vspace{4pt}%
  \@bsfield{\authorname}{\ThesisAuthor}%
  \@bsfield{\thesisname}{\th@sistitl@}%
  \@bsfield{\degreeordep@rtmentname}{\degreeordep@rtment}%
  \noindent%
  \parbox[t]{\fnamelenA}{\mytextsf{\majorname}}%
  \hspace{1ex}%
  \parbox[t]{\abstractwidth-\fnamelenA-\fnamelenB-\ftextlen-3ex}{%
	\raggedright\m@jor{}%
  }%
  \hfill%
  \parbox[t]{\fnamelenB}{\mytextsf{\codename}}%
  \hspace{1ex}%
  \parbox[t]{\ftextlen}{\Cod@}%
  \@bsbre@k%
  \ifnumequal{\value{advisors}}{0}{%
	\@bsfield{\supervisorname}{\sup@rvis@r}%
  }{%
	\@bsfield{\supervisorname}{\sup@rvis@r}%
	\@bsfield{\advisorname}{\Advis@r{1}%
	\ifnumgreater{\value{advisors}}{1}{, \Advis@r{2}}{}}%
  }%
  \noindent%
  \parbox[t]{\abstractwidth}{%
	\mytextsf{\datename}\hspace{1ex}\D@te{}\hfill%
	\mytextsf{\nofpgsname}\hspace{1ex}\shownumberofpages%
	\hfill\mytextsf{\langname}\hspace{1ex}\L@ng{#1}%
  }%
  \@bsbre@k%
  \vspace{2pt}%
%
  \noindent\mytextsf{\abstractname}\\%
  \ifdefstrequal{\MainLang}{\AbstractLang}{}{\begin{otherlanguage*}{#1}}%
}{%
  \ifdefstrequal{\MainLang}{\AbstractLang}{}{\end{otherlanguage*}}%
  \vspace{2pt}%
  \@bsbre@k%
  \@bsfield{\keywordsname}{\k@ywords\par\vspace{0.3ex}}%
  \setcounter{advisors}{0}%
  \clearpage%
}%
%%%%%% Abstract stuff ends here %%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing the metadata
\newcommand{\makemetadata}{%
  \iftoggle{writexmpdatafile}{%
    % Write the metadata into file .xmpdat, which is included in the PDF/A file
	\newoutputstream{xmpdata}%
	\openoutputfile{\jobname.xmpdata}{xmpdata}%
	\addtostream{xmpdata}{\noexpand\Title {\pdfth@sistitl@}}%
	\addtostream{xmpdata}{\noexpand\Author {\ThesisAuthor}}%
	\addtostream{xmpdata}{\noexpand\Copyright {\MetaCopyrightText}}%
	\addtostream{xmpdata}{\noexpand\Subject {\abstracttext}}%
	\addtostream{xmpdata}{\noexpand\Keywords {\k@ywords}}%
	\closeoutputstream{xmpdata}%
  }{%
    % Write the metadata directly into the pdf file via hyperref keywords
	\renewcommand*{\spc}{; }%
	\hypersetup{%
	  pdftitle={\pdfth@sistitl@},%
	  pdfauthor={\ThesisAuthor},%
	  pdfsubject={\abstracttext},%
	  pdfcopyright={\CopyrightText},%
	  pdfkeywords={\k@ywords}%
	}%
  }%
}%
%
\AtBeginDocument{\makemetadata}%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A few more macros
%%
%% \phantomsection is the command used by hyperref to set a bookmark to 
%% starred sections so that links have the right target, e.g. in a toc.
%% \phantomsection is a dummy command without the hyperref package.
%\AtBeginDocument{\providecommand*{\phantomsection}{}}%
%
\newcommand*{\mysection}[1]{%
  \newpage%
  \section*{#1}%
  \addcontentsline{toc}{section}{#1}%
}%
%
\newcommand{\thesistableofcontents}{%
  \phantomsection%
  \addcontentsline{toc}{section}{\contentsname}%
  \tableofcontents%
}%
%
\newcommand{\thesisappendix}{%
  \stor@maintxtpag@number%
  \appendix%
  \counterwithin{equation}{section}%
  \renewcommand{\theequation}{\thesection\arabic{equation}}%
  \counterwithin{figure}{section}%
  \renewcommand{\thefigure}{\thesection\arabic{figure}}%
  \renewcommand{\theHfigure}{\thesection\arabic{figure}}% for hyperref
  \counterwithin{table}{section}%
  \renewcommand{\thetable}{\thesection\arabic{table}}%
  \renewcommand{\theHtable}{\thesection\arabic{table}}% for hyperref
}%
%
\newcommand*{\thesisbibliography}{%
  \phantomsection%
  \addcontentsline{toc}{section}{\refname}%
}%
%
\AtBeginDocument{\pagestyle{myheadings}}%
%
\endinput
