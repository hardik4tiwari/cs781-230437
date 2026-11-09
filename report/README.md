# Report source and completion checklist

`Report.tex` follows the CS781 report template's title block and eight numbered sections. Phase 7 is included because it is in the course brief. The main report has an eight-page limit, excluding references and the LLM declaration. The draft is a record of available evidence.

Compile from this directory with TeX Live or MacTeX.

```sh
latexmk -pdf Report.tex
```

From the repository root, use the archive builder after compiling.

```sh
python3 build_submission.py report/Report.pdf --draft
```

Omit `--draft` after completing the checklist to produce the exact Hello IITK archive and internal PDF filenames.

## Complete before submitting

- Fill in the IITK email, Codabench username, leaderboard ranks, rank-freeze date and time zone, and official valid submission counts. The 44 visible finished rows are an evidence subtotal. The course brief calls for four distinct submissions in Phase 1 and six per phase in Phases 2–7.
- Add Phase 6's actual run ID, measured metrics, candidate ZIPs, and official result. Record whether the candidates came from the repaired full suite or the fast notebook. The two variants use overlapping candidate IDs.
- Confirm the checkpoint-to-upload mapping for the best Codabench rows. The local Phase 5 archive and the full-document notebook agree on both observed aggregate counts. A byte-for-byte match to Codabench needs a payload or reference hash from that upload.
- Use `cs781-legal-intelligence` as the GitHub repository name and add its public URL after publishing the reviewed source and declaration. This repository is local.
- Finish the abstract's findings, results interpretation, controlled-comparison discussion, error analysis, and conclusions in your own words. Add actual example IDs and counts where relevant and verify the bibliography against the sources you used.
- Complete and sign off on the LLM usage declaration. It records visible assistance from the reference chat as a draft. Confirm its scope and list every tool and use, including any that fall outside the course's permitted categories.
- Remove the draft notice and editing prompts, recompile, and check the page count and archive before uploading.

See [the official CS781 LaTeX template](https://drive.google.com/file/d/1T9HVUeT2NerQTJD8A--tGgO-tDN1ww40/view) and [the evidence index](evidence/README.md).
