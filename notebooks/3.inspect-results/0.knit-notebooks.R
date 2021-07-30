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

# --------

dataset <- "Stain5_CondC_PE_Standard"
metadata <-
  list(
    r_start = 2L,
    r_end = 3L,
    c_start = 5L,
    c_end = 6L,
    f_start = 8L,
    f_end = 9L,
    ch_start = 16L,
    ch_end = 16L
  )
split_string <-
  c("x1", "x2", "x3", "project", "batch", "x", "plate", "x5", "image")

render_notebook(
  "1.inspect-test-structure",
  output_suffix = paste0("_", dataset),
  params = list(
    dataset = dataset,
    metadata = metadata,
    split_string = split_string
  )
)

render_notebook(
  "2.inspect-test-results",
  output_suffix = paste0("_", dataset),
  params = list(dataset = dataset)
)

# --------
dataset <- "Plate3_PCO_6ch_4site_10XPA_Crestz"
metadata <-
  list(
    r_start = 36L,
    r_end = 36L,
    c_start = 37L,
    c_end = 38L,
    f_start = 41L,
    f_end = 41L,
    ch_start = 44L,
    ch_end = 44L
  )
split_string <-
  c("x1", "x2", "x3", "project", "batch", "x", "plate", "x5", "image")

render_notebook(
  "1.inspect-test-structure",
  output_suffix = paste0("_", dataset),
  params = list(
    dataset = dataset,
    metadata = metadata,
    split_string = split_string
  )
)

render_notebook(
  "2.inspect-test-results",
  output_suffix = paste0("_", dataset),
  params = list(dataset = dataset)
)
