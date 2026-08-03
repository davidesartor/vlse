## The (SFU R name, function) table tools/eval_r.R drives the parity test with.

sfu_fns <- function(sfu) {
  for (path in list.files(sfu, pattern = "\\.R$", full.names = TRUE)) {
    sys.source(path, envir = globalenv())
  }

  langA <- matrix(c(3, 5, 5, 2, 2, 1, 1, 4, 7, 9), 5, 2, byrow = TRUE)
  psum_b <- function(d) sapply(1:d, function(i) sum((1:d)^i))
  # forretal08lc.R source()s a file name we do not have; this is its body
  forretal08lc <- function(x, A = 0.5, B = 10, C = -5) A * forretal08(x) + B * (x - 0.5) - C

  list(
    ackley = ackley, bukin6 = bukin6, crossit = crossit, drop = drop, egg = egg,
    grlee12 = grlee12, griewank = griewank, holder = holder,
    langer = function(x) langer(x, m = 5, cvec = c(1, 2, 5, 2, 3), A = langA),
    levy = levy, levy13 = levy13, rastr = rastr, schaffer2 = schaffer2,
    schaffer4 = schaffer4, schwef = schwef, shubert = shubert,
    boha1 = boha1, boha2 = boha2, boha3 = boha3,
    perm0db = function(x) perm0db(x, b = 10), rothyp = rothyp, spheref = spheref,
    spherefmod = spherefmod,
    sumpow = sumpow, sumsqu = sumsqu, trid = trid, booth = booth, matya = matya,
    mccorm = mccorm, powersum = function(x) powersum(x, b = psum_b(length(x))),
    zakharov = zakharov, camel3 = camel3, camel6 = camel6, dixonpr = dixonpr,
    rosen = rosen, rosensc = rosensc, dejong5 = dejong5, easom = easom,
    michal = function(x) michal(x, m = 10), beale = beale, branin = branin,
    braninsc = braninsc, braninmodif = braninmodif,
    colville = colville, forretal08 = forretal08, forretal08lc = forretal08lc,
    goldpr = goldpr, goldprsc = goldprsc,
    hart3 = hart3, hart4 = hart4, hart6 = hart6, hart6sc = hart6sc,
    permdb = function(x) permdb(x, b = 10), powell = powell,
    shekel5 = function(x) shekel(x, m = 5), shekel7 = function(x) shekel(x, m = 7),
    shekel10 = function(x) shekel(x, m = 10), stybtang = stybtang
  )
}
