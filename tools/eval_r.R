## Evaluate the SFU R implementations on <work>/x/<name>.csv, writing <work>/y/<name>.csv.

args <- commandArgs(trailingOnly = TRUE)
sfu <- args[1]
work <- args[2]
source(file.path(sfu, "..", "fns_r.R"))
FNS <- sfu_fns(sfu)

dir.create(file.path(work, "y"), showWarnings = FALSE)
for (name in names(FNS)) {
  X <- as.matrix(read.csv(file.path(work, "x", paste0(name, ".csv")), header = FALSE))
  y <- apply(X, 1, FNS[[name]])
  write.table(y, file.path(work, "y", paste0(name, ".csv")), row.names = FALSE, col.names = FALSE)
}
cat("R done:", length(FNS), "functions\n")
