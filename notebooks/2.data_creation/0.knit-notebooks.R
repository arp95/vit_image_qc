output_format <- "github_document"

render_notebook <-
  function(notebook_name, output_suffix = "", ...) {
    output_file <- paste0(notebook_name, output_suffix, ".md")

    rmarkdown::render(
      glue::glue("{notebook_name}.Rmd"),
      output_file = output_file,
      output_dir = "knit_notebooks",
      output_format = output_format,
      ...
    )

    output_file_rel <- file.path("knit_notebooks", output_file)

    read_lines(output_file_rel) %>%
      str_remove_all(file.path(getwd(), "knit_notebooks/")) %>%
      write_lines(output_file_rel)

  }

render_notebook("2.parse_training")

render_notebook("3.inspect-annotations")

render_notebook("4.plot-annotations")

render_notebook("5.parse-partner-images")

render_notebook("6.inspect-train-test")

render_notebook("7.cellprofiler-feature-matrix")

render_notebook("8.inspect-features")
