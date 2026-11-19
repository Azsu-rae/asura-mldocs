# /bin/bash

set -e

pdflatex -output-directory=tex $1
pdf2svg tex/$2.pdf tex/nn/$2.svg
xdg-open tex/nn/$2.svg

rm tex/$2.pdf
