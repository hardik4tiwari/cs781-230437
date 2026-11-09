# Report source

[Report.tex](Report.tex) is the editable LaTeX source. Keep the evidence directory beside it because the two learning curves read their data from that directory. The title and numbered sections follow the course report template.

Compile from the report directory with TeX Live or MacTeX.

~~~sh
latexmk -pdf Report.tex
~~~

From the repository root, rebuild the current package with the report and seven phase directories.

~~~sh
python3 build_submission.py report/Report.pdf
~~~

## Before course submission

- The report records all seven supplied ranks and 48 visible finished rows. Four Phase 6 prediction sets were verified as distinct. The attached template lists a four-submission minimum, while the capstone brief specifies six distinct downstream submissions. The Phase 5 generic upload matches the extra notebook's aggregate prediction counts. A Codabench payload hash is unavailable.
- Use the documented measurements to write the student-authored analysis and conclusions required by the course. The declaration includes the student’s account of their contributions alongside the LLM assistance recorded in the project history.
- Recompile and check the compiled PDF and archive before uploading.

See the [evidence index](evidence/README.md) and [GitHub repository](https://github.com/hardik4tiwari/cs781-230437).
