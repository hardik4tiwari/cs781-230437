# Report source

[Report.tex](Report.tex) is the editable LaTeX source. Keep the evidence directory beside it because the two learning curves read their data from that directory. The title and numbered sections follow the course report template.

Compile from the report directory with TeX Live or MacTeX.

~~~sh
latexmk -pdf Report.tex
~~~

From the repository root, rebuild the current package with the report and seven phase directories.

~~~sh
python3 build_submission.py report/Report.pdf --draft
~~~

## Before course submission

- Verify the leaderboard ranks and distinct valid submission counts. Phase 6 had not been submitted when this report was prepared. The 44 visible finished rows are a subtotal. The Phase 5 generic upload matches the extra notebook's aggregate prediction counts. A Codabench payload hash is unavailable.
- Use the documented measurements to write the student-authored analysis and conclusions required by the course. The declaration includes the student’s account of their contributions alongside the LLM assistance recorded in the project history.
- Recompile and omit the draft option when the report meets the course requirements. Check the compiled PDF and the archive before uploading.

See the [evidence index](evidence/README.md) and [GitHub repository](https://github.com/hardik4tiwari/cs781-230437).
