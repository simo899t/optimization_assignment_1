#import "@preview/wordometer:0.1.5": word-count as _word-count, total-words as totalwords
#let word-count = _word-count
#let total-words = totalwords
#import "@preview/lovelace:0.3.0": *
#import "@preview/tdtr:0.5.2" : *
#import "@preview/h-graph:0.1.0": *
#import "@preview/cetz:0.3.4": canvas, draw
// Proof trees
// r(label, formula)                    — axiom leaf
// r(label, formula, r(...), r(...))    — sub-derivation (nests automatically)
// ptree(conclusion, r(...), r(...))    — root inference step
#let r(name, formula, ..subs) = (name, formula, ..subs.pos())

#let ptree(
  conclusion,
  gap: auto,       // horizontal gap between premises
  above: auto,     // space above the line
  below: auto,     // space below the line
  label-gap: auto, // space between label and formula
  pad: auto,       // extra line width on each side
  inset: auto,     // bottom padding (pushes caption down in figures)
  ..args,
) = context {
  let em = measure(line(length: 1em)).width
  let gap       = if gap       == auto { 4 * em  } else { gap }
  let above     = if above     == auto { 0.5 * em } else { above }
  let below     = if below     == auto { 0.5 * em } else { below }
  let label-gap = if label-gap == auto { 0.6 * em } else { label-gap }
  let pad       = if pad       == auto { 1.5 * em } else { pad }
  let inset     = if inset     == auto { 1 * em   } else { inset }
  let prems = args.pos()
  if prems.len() == 0 { return conclusion }

  let render-prem(p) = if type(p) == array {
    if p.len() > 2 {
      // sub-derivation: recurse, then put label above the sub-tree
      let sub = ptree(p.at(1), gap: gap, above: above, below: below,
                      label-gap: label-gap, pad: pad, ..p.slice(2))
      align(center, stack(dir: ttb, spacing: label-gap, p.at(0), sub))
    } else {
      align(center, stack(dir: ttb, spacing: label-gap, p.at(0), p.at(1)))
    }
  } else { align(center, p) }

  let prem-row = grid(
    columns: (auto,) * prems.len(),
    column-gutter: gap,
    ..prems.map(render-prem)
  )

  let lw = calc.max(measure(prem-row).width, measure(conclusion).width) + pad

  align(center, stack(dir: ttb, spacing: 0pt,
    align(center, prem-row),
    v(above),
    line(length: lw),
    v(below),
    align(center, conclusion),
    v(inset),
  ))
}
#let dirgraph(src) = h-graph(src, polar-render)

#let base-style(body) = {
  show: _word-count
  set text(font: "Times New Roman", size: 11pt, lang: "en")
  set heading(numbering: "1.")
  set math.equation(numbering: none)
  set math.mat(delim: "[", gap: 0.3em)
  set par(justify: true)
  set image(width: 30em)
  body
}
#let bib = bibliography.with(style: "chicago-author-date")

// Applied globally so docs without a template also get base styling.
// Templates call base-style(body) internally and set their own page rules.
#set page(paper: "us-letter", margin: (left: 3cm, right: 3cm, top: 2cm, bottom: 2cm))
#show: base-style

// syntax formatting
#let hs(x) = raw(x, lang: "hs")
#let py(x) = raw(x, lang: "py")


// just nice
#let abc = enum.with(numbering: "(a)", spacing: 1.5em)
#let evaluated(expr, size: 100%) = $lr(#expr|, size: #size)$
#let u(x) = underline(x)
#let b(x) = bold(x)
#let i(x) = emph[x]
#let yes = $checkmark$
#let no = $crossmark$
#let absurd = $bot$
#let all(p,q) = $"All" #p "are" #q$
#let def = $=^"def"$
#let supremum(x) = $op("supremum", limits: #true)_(#x)$
#let softmax(x) = $"softmax"(#x)$
#let wrt = $w.r.t$
#let bigo(x) = $cal(O)(#x)$
#let smallo(x) = $cal(o)(#x)$
#let ie = $i.e.$
#let eg = $e.g.$
#let Der = "Der"
#let ip(x) = $[|#x|]$
// shortcuts
#let redmath(x) = text(fill: red, $#x$)
#let bluemath(x) = text(fill: blue, $#x$)
#let greenmath(x) = text(fill: green, $#x$)
#let int(a,b,c) = $integral_(#a)^(#b) #c$ 
#let prod(a,b,c) = $product_(#a)^(#b) #c$ 
#let summ(a,b,c) = $sum_(#a)^(#b) #c$ 
#let limm(a) = $lim_(#a)$
#let pred(a) = $accent(#a,\^)$
#let QED = [#h(1fr) $square$]
#let IH = [*_IH_*]
#let f = [#h(1fr)]
#let qquad = $quad quad$
#let qqquad = $quad quad quad$
#let qqqquad = $quad quad quad quad$
#let sign(a) = $"sign"(#a)$
#let psubset = $subset.eq$
#let rang = $chevron.r$
#let lang = $chevron.l$
#let pow(x) = $cal(P)(#x)$

// --- Calculus notation ---
#let dx = $dif x$
#let dy = $dif y$
#let dz = $dif z$
#let dx() = $dif #x$
#let px() = $partial #x$

// Ordinary derivatives
#let ddx = $dif/(dif x$                         // d/dx  (operator)
#let ddy = $dif/(dif y$                         // d/dy  (operator)
#let ddz = $dif/(dif z$                         // d/dz  (operator)
#let dd(x) = $dif/(dif #x)$                     // d/d(var)  e.g. dd(t)
#let dv(f, x) = $(dif #f)/(dif #x)$             // df/dx  e.g. dv(f,x)
#let dvn(f, x, n) = $(dif^#n #f)/(dif #x^#n)$   // dⁿf/dxⁿ  e.g. dvn(f,x,2)

// Partial derivatives
#let ppx = $partial/(partial x)$ 
#let ppy = $partial/(partial y)$    
#let ppz = $partial/(partial z)$            // ∂/∂x  (operator)
#let pp(x) = $partial/(partial #x)$                           // ∂/∂(var)  e.g. pp(y)
#let pv(f, x) = $(partial #f)/(partial #x)$                   // ∂f/∂x  e.g. pv(f,x)
#let pvn(f, x, n) = $(partial^#n #f)/(partial #x^#n)$        // ∂ⁿf/∂xⁿ  e.g. pvn(f,x,2)
#let pvm(f, x, y) = $(partial^2 #f)/(partial #x partial #y)$ // ∂²f/∂x∂y  mixed partial

// Hessian (2×2 and n×n pattern)
#let hess2(f) = $mat(
  (partial^2 #f)/(partial x^2), (partial^2 #f)/(partial x partial y);
  (partial^2 #f)/(partial y partial x), (partial^2 #f)/(partial y^2)
)$
#let hess(f) = $mat(
  (partial^2 #f)/(partial x_1^2), dots.c, (partial^2 #f)/(partial x_1 partial x_n);
  dots.v, dots.down, dots.v;
  (partial^2 #f)/(partial x_n partial x_1), dots.c, (partial^2 #f)/(partial x_n^2)
)$

// Gradient / Laplacian
#let nf(f) = $nabla #f$
#let nnf(f) = $nabla^2 #f$
#let nf = $nabla f$

// pseudocode alias
#let pseudo = pseudocode-list

// symbols
#let phi = $phi.alt$
#let eps = $epsilon$
#let Eps = $Epsilon$
#let del = $delta$
#let Del = $Delta$
#let gam = $gamma$
#let Gam = $Gamma$
#let cap = $inter$
#let cup = $union$
#let ent = symbol("⊨", ("not", "⊭"))
#let prov = symbol("⊢", ("not", "⊬"))
#let model = $cal(M)$
#let apx = $approx$
#let dag = $dagger$


// --- Subset diagram (Euler / nested-circles) ---
// Draws a chain of nested circles: outermost set first, innermost last.
//
// sets     — array of set specs, each one of:
//              "key"                    key used in elements dict, displayed as-is
//              ("key", label)           string key + any content as display label
//              ("key", label, color)    as above with explicit fill colour
//            Omit entirely to auto-derive from elements dict keys.
// elements — dict  key → array of content (any content: strings, math…)
// gap      — radial distance between consecutive circles (cm)
//
// Usage:
//   #subset-diagram(elements: ("A": ("1","2"), "B": ("3","4","5")))
//   #subset-diagram(
//     sets: (("A", $cal(A)$), ("B", $cal(B)$)),
//     elements: ("A": ($[|a|]$,), "B": ($x^2$, $y$)),
//   )
#let subset-diagram(sets: none, elements: (:), gap: 1.0) = {
  let sets = if sets == none { elements.keys() } else { sets }
  let defaults = (
    rgb(214, 234, 248, 180), // sky blue
    rgb(212, 239, 223, 180), // mint green
    rgb(253, 235, 208, 180), // peach
    rgb(245, 215, 215, 180), // rose
    rgb(235, 218, 255, 180), // lavender
    rgb(255, 245, 200, 180), // butter yellow
    rgb(200, 240, 240, 180), // aqua
    rgb(255, 220, 200, 180), // apricot
    rgb(220, 220, 255, 180), // periwinkle
    rgb(210, 245, 210, 180), // sage
  )
  canvas({
    import draw: *
    let n = sets.len()
    let base-r = n * gap

    for i in range(n) {
      let spec = sets.at(i)
      let (key, lbl, col) = if type(spec) == array and spec.len() >= 3 {
        (spec.at(0), spec.at(1), spec.at(2))
      } else if type(spec) == array and spec.len() == 2 {
        (spec.at(0), spec.at(1), defaults.at(calc.rem(i, defaults.len())))
      } else {
        (spec, spec, defaults.at(calc.rem(i, defaults.len())))
      }
      let r = base-r - i * gap

      circle((0, 0), radius: r,
        stroke: 1pt + rgb(80, 100, 140),
        fill: col,
        name: "s" + str(i),
      )
      content(
        (-r * 0.72, r * 0.82),
        text(size: 9pt, weight: "bold", lbl),
      )

      // elements placed evenly around the mid-radius of this ring
      if key in elements {
        let raw     = elements.at(key)
        let elems   = if type(raw) == array { raw } else { (raw,) }
        let inner-r = if i + 1 < n { base-r - (i + 1) * gap } else { 0 }
        let mid-r   = (r + inner-r) / 2 * 0.75
        let count   = elems.len()
        for j in range(count) {
          let angle = 360deg / count * j - 90deg
          content(
            (mid-r * calc.cos(angle), mid-r * calc.sin(angle)),
            text(size: 8pt, elems.at(j)),
          )
        }
      }
    }
  })
}

// --- Euler diagram (general overlapping/nested ellipses) ---
// Use this when sets are not a simple chain (siblings, overlaps, lattices).
//
// sets     — array of (key, label, cx, cy, rx, ry) or (…, color)
//            listed in any order — drawn largest-area-first automatically
// elements — array of (display, (key1, key2, …))
//            each element is placed at the centre of its smallest containing set;
//            multiple elements sharing the same tightest container are spread out
//
// Usage:
//   #euler-diagram(
//     sets: (
//       ("P",  $P$,      0,    0,    3.5, 2.5),
//       ("b",  $[|b|]$, -2,   0,    1.0, 0.7),
//       ("d",  $[|d|]$,  0.5,  0.5, 1.5, 1.1),
//       ("e",  $[|e|]$,  1.2, -0.5, 1.5, 1.1),
//       ("c",  $[|c|]$,  0.85, 0,   0.85, 0.65),
//       ("a",  $[|a|]$,  0.85, 0,   0.32, 0.28),
//     ),
//     elements: (
//       ($a$, ("a",)),
//       ($b$, ("b",)),
//       ($c$, ("c",)),
//       ($d$, ("d",)),
//       ($e$, ("e",)),
//     ),
//   )
// Euler diagram — blobs morph to fit elements.
// sets:     array of (key, label). First set = universe, auto-contains all elements.
// elements: array of (display, (key1, key2, ...), (x, y)).
//           Position (x, y) places the element; the blob for every set it belongs
//           to automatically expands to encapsulate it.
// padding:  extra space around elements inside each blob.
#let euler-diagram(sets, elements: (), padding: 0.8) = {
  let defaults = (
    rgb(214, 234, 248, 150), // sky blue
    rgb(212, 239, 223, 150), // mint green
    rgb(253, 235, 208, 150), // peach
    rgb(245, 215, 215, 150), // rose
    rgb(235, 218, 255, 150), // lavender
    rgb(255, 245, 200, 150), // butter yellow
    rgb(200, 240, 240, 150), // aqua
    rgb(255, 220, 200, 150), // apricot
    rgb(220, 220, 255, 150), // periwinkle
    rgb(210, 245, 210, 150), // sage
  )

  let universe-key = sets.at(0).at(0)

  // Read element positions from the 3rd field if present, else (0,0).
  let elem-data = elements.map(item => {
    let (ex, ey) = if item.len() >= 3 {
      (float(item.at(2).at(0)), float(item.at(2).at(1)))
    } else { (0.0, 0.0) }
    (item.at(0), ex, ey, item.at(1))
    // (display, x, y, keys)
  })

  // Collect element positions for each set.
  // Universe always gets all elements; sub-sets get elements that list them.
  let set-pts = (:)
  for s in sets { set-pts = set-pts + ((s.at(0)): ()) }
  for ed in elem-data {
    let pos = (ed.at(1), ed.at(2))
    set-pts = set-pts + ((universe-key): set-pts.at(universe-key) + (pos,))
    for k in ed.at(3) {
      if k != universe-key and k in set-pts {
        set-pts = set-pts + ((k): set-pts.at(k) + (pos,))
      }
    }
  }

  // Draw order: most elements first (background → foreground).
  let ordered = sets.map(s => s.at(0)).sorted(key: k => -set-pts.at(k).len())

  canvas({
    import draw: *

    for (idx, key) in ordered.enumerate() {
      let s   = sets.filter(s => s.at(0) == key).at(0)
      let lbl = s.at(1)
      let col = defaults.at(calc.rem(idx, defaults.len()))
      let pts = set-pts.at(key)

      // Centroid of this set's element positions.
      let (cx, cy) = if pts.len() == 0 { (0.0, 0.0) } else {
        ( pts.map(p => p.at(0)).sum() / float(pts.len()),
          pts.map(p => p.at(1)).sum() / float(pts.len()) )
      }

      // Blob boundary via support function:
      // r(α) = padding + max projection of any element onto direction α.
      let n-pts = 60
      let blob = range(n-pts).map(j => {
        let alpha = 360deg * float(j) / float(n-pts)
        let ca = calc.cos(alpha)
        let sa = calc.sin(alpha)
        let r = pts.fold(padding, (acc, p) => {
          let proj = (p.at(0) - cx) * ca + (p.at(1) - cy) * sa
          if proj > 0 { calc.max(acc, proj + padding) } else { acc }
        })
        (cx + r * ca, cy + r * sa)
      })

      catmull(..blob, close: true, fill: col, stroke: 1pt + rgb(80, 100, 140))

      // Label at upper-left of the blob.
      let lca = calc.cos(135deg)
      let lsa = calc.sin(135deg)
      let lr  = pts.fold(padding + 0.25, (acc, p) => {
        let proj = (p.at(0) - cx) * lca + (p.at(1) - cy) * lsa
        if proj > 0 { calc.max(acc, proj + padding + 0.25) } else { acc }
      })
      content((cx + lr * lca, cy + lr * lsa),
        text(size: 9pt, weight: "bold", lbl))
    }

    // Render elements at their computed positions.
    for ed in elem-data {
      content((ed.at(1), ed.at(2)), text(size: 8pt, ed.at(0)))
    }
  })
}

// --- Venn / automatic Euler diagram ---
// Pass set specs as positional args: (label, (elem1, elem2, ...))
// Elements are strings used for layout; labels are any content.
// domain: optional label for the universe ellipse.
// scale:  spacing between elements (cm).
// pad:    extra padding around each ellipse.
//
// Usage:
//   #venn(
//     domain: $P$,
//     ($[|a|]$, ("a",)),
//     ($[|b|]$, ("a","b")),
//     ($[|c|]$, ("a","c")),
//     ($[|d|]$, ("a","c","d")),
//     ($[|e|]$, ("a","c","e")),
//   )
#let venn(domain: none, scale: 1.4, pad: 0.6, ..args) = {
  let sets = args.pos()
  let defaults = (
    rgb(214, 234, 248, 150),
    rgb(212, 239, 223, 150),
    rgb(253, 235, 208, 150),
    rgb(245, 215, 215, 150),
    rgb(235, 218, 255, 150),
    rgb(255, 245, 200, 150),
    rgb(200, 240, 240, 150),
    rgb(255, 220, 200, 150),
    rgb(220, 220, 255, 150),
    rgb(210, 245, 210, 150),
  )

  // Count how many sets each element appears in
  let freq = (:)
  for (_, elems) in sets {
    for e in elems {
      freq = freq + ((e): (if e in freq { freq.at(e) } else { 0 }) + 1)
    }
  }

  // Place elements on a golden-angle spiral: most frequent = closest to centre
  let sorted-elems = freq.keys().sorted(key: e => -freq.at(e))
  let pos = (:)
  for (i, e) in sorted-elems.enumerate() {
    let r     = if i == 0 { 0.0 } else { scale * calc.sqrt(float(i)) }
    let angle = float(i) * 2.3999632rad
    pos.insert(e, (r * calc.cos(angle), r * calc.sin(angle)))
  }

  // Compute bounding ellipse (cx, cy, rx, ry) for a list of element keys
  let bbox(elems) = {
    let xs = elems.map(e => pos.at(e).at(0))
    let ys = elems.map(e => pos.at(e).at(1))
    let xmin = xs.fold(xs.at(0), calc.min)
    let xmax = xs.fold(xs.at(0), calc.max)
    let ymin = ys.fold(ys.at(0), calc.min)
    let ymax = ys.fold(ys.at(0), calc.max)
    let cx = (xmin + xmax) / 2
    let cy = (ymin + ymax) / 2
    let rx = calc.max((xmax - xmin) / 2 + pad, pad * 0.7)
    let ry = calc.max((ymax - ymin) / 2 + pad, pad * 0.7)
    (cx, cy, rx, ry)
  }

  canvas({
    import draw: *

    // Domain ellipse (outermost)
    if domain != none {
      let all-keys = freq.keys()
      let (cx, cy, rx, ry) = bbox(all-keys)
      let rx2 = rx + pad * 0.6
      let ry2 = ry + pad * 0.6
      circle((cx, cy), radius: (rx2, ry2),
        fill: rgb(230, 230, 230, 100), stroke: 1pt + rgb(80, 100, 140))
      content((cx - rx2 * 0.78, cy + ry2 * 0.84),
        text(size: 9pt, weight: "bold", domain))
    }

    // Draw set ellipses, largest first so smaller ones appear on top
    let order = range(sets.len()).sorted(key: i => -sets.at(i).at(1).len())
    for i in order {
      let (lbl, elems) = (sets.at(i).at(0), sets.at(i).at(1))
      if elems.len() == 0 { continue }
      let (cx, cy, rx, ry) = bbox(elems)
      let col = defaults.at(calc.rem(i, defaults.len()))
      circle((cx, cy), radius: (rx, ry),
        fill: col, stroke: 1pt + rgb(80, 100, 140))
      content((cx - rx * 0.72, cy + ry * 0.82),
        text(size: 9pt, weight: "bold", lbl))
    }

    // Element labels at their computed positions
    for (e, p) in pos {
      content(p, text(size: 8pt, e))
    }
  })
}

#let tree(body, reverse: false, shape: "circle", draw-node: none, ..args) = {
  let shape-draw-node = if shape == "circle" {
    tidy-tree-draws.circle-draw-node
  } else if shape == "rect" or shape == "rectangle" {
    ((name, label, pos)) => (shape: rect)
  } else if shape == "square" {
    ((name, label, pos)) => (shape: rect, width: 1.6em, height: 1.6em)
  } else {
    tidy-tree-draws.circle-draw-node
  }
  let effective-draw-node = if draw-node != none { draw-node } else { shape-draw-node }
  let draw-nodes = if reverse {
    (effective-draw-node, ((name, label, pos)) => (pos: (pos.x, -pos.y)))
  } else {
    effective-draw-node
  }
  tidy-tree-graph(body, draw-node: draw-nodes, ..args)
}


#let group-by-pairs(elements) = {
  let lefts = elements
    .enumerate()
    .filter(((index, _)) => calc.rem(index, 2) == 0)
    .map(((_, element)) => element)
  let rights = elements
    .enumerate()
    .filter(((index, _)) => calc.rem(index, 2) == 1)
    .map(((_, element)) => element)
  lefts.zip(rights)
}

#let mycases(..cases, word: none) = {
  let cases = group-by-pairs(cases.pos())
    .map(((value, condition)) => {
      if word != none {
        $#value quad &#word #condition$
      } else {
        $#value quad & #condition$
      }
    })
  math.cases(..cases)
}

// Helper: accepts a string or array of strings, formats as "A · B · C"
#let _fmt-authors(author) = {
  if type(author) == str { author }
  else { author.join(" · ") }
}

#let code(content) = block(
  width: 100%,
  fill: rgb("#f8f9fa"),
  stroke: (left: 2.5pt + rgb("#4a7fc1"), rest: 0.5pt + rgb("#d8dde6")),
  inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
  radius: 2pt,
  [
    #set par(leading: 0.8em)
    #text(
      fill: rgb("#1c1e26"),
      font: "JetBrains Mono",
      size: 9.5pt,
      weight: "regular",
    )[#content]
  ]
)

#let theorem(title: "Theorem", content) = block(
  fill: gradient.linear(
    rgb("#fafbfc"), 
    rgb("#f1f3f4"), 
    angle: 135deg
  ),
  stroke: (
    left: 3pt + rgb("#2c5aa0"),
    rest: 0.5pt + rgb("#e1e5e9")
  ),
  inset: (left: 18pt, right: 14pt, top: 14pt, bottom: 14pt),
  radius: 8pt,
  [
    #text(weight: "bold", fill: rgb("#1a365d"), size: 12.5pt)[#title]
    #v(0.5em)
    #text(fill: rgb("#2d3748"), size: 10.5pt)[#content]
  ]
)

#let definition(title: "Definition", content) = block(
  fill: gradient.linear(
    rgb("#fffef7"), 
    rgb("#fef9e7"), 
    angle: 135deg
  ),
  stroke: (
    left: 3pt + rgb("#d69e2e"),
    rest: 0.5pt + rgb("#f7d794")
  ),
  inset: (left: 18pt, right: 14pt, top: 14pt, bottom: 14pt),
  radius: 8pt,
  [
    #text(weight: "bold", fill: rgb("#744210"), size: 12.5pt)[#title]
    #v(0.5em)
    #text(fill: rgb("#553c0f"), size: 10.5pt)[#content]
  ]
)

// --- Document metadata (override in your file) ---


#let default-title = "Untitled Document"
#let default-course= "SDU"
#let default-author = "Simon Holm"
#let default-date = "16/12/2002"


#let note(
  title: default-title,
  author: default-author,
  course: default-course,
  date: default-date,
  outline: true,
  outline-depth: none,
  ..args,
) = {
  let body = args.pos().at(0, default: [])
  set page(paper: "us-letter", margin: (left: 3cm, right: 3cm, top: 3cm, bottom: 3cm))
  align(center,
    stack(
      spacing: 0pt,
      v(1.2cm),
      // Blue top bar + label
      line(length: 100%, stroke: 3pt + rgb("#2c5aa0")),
      v(1.2em),
      text(size: 9.5pt, fill: rgb("#2c5aa0"), tracking: 2.5pt, weight: "bold")[LECTURE NOTES],
      v(2.5cm),
      // Title
      text(size: 30pt, weight: "bold")[#title],
      v(1.3em),
      line(length: 28%, stroke: 0.5pt + rgb("#bbbbbb")),
      v(0.7em),
      text(size: 14pt, fill: rgb("#444444"))[#course],
      // Push to bottom
      v(1fr),
      text(size: 12pt)[#_fmt-authors(author)],
      v(0.3em),
      text(size: 11pt, fill: rgb("#888888"))[#date],
      v(1.8em),
      image("IMADA_en.png", width: 15em),
      v(1cm),
    )
  )
  pagebreak()
  if outline { std.outline(depth: outline-depth); pagebreak() }
  base-style(body)
}

#let exercise(
  title: default-title,
  author: default-author,
  course: default-course,
  date: default-date,
  outline: true,
  outline-depth: none,
  ..args,
) = {
  let body = args.pos().at(0, default: [])
  set page(paper: "us-letter", margin: (left: 3cm, right: 3cm, top: 3cm, bottom: 3cm))
  align(center,
    stack(
      spacing: 0pt,
      v(1.2cm),
      // Amber top bar + label
      line(length: 100%, stroke: 3pt + rgb("#b7410e")),
      v(1.2em),
      text(size: 9.5pt, fill: rgb("#b7410e"), tracking: 2.5pt, weight: "bold")[EXERCISES],
      v(2.5cm),
      // Title
      text(size: 30pt, weight: "bold")[#title],
      v(1.3em),
      line(length: 28%, stroke: 0.5pt + rgb("#bbbbbb")),
      v(0.7em),
      text(size: 14pt, fill: rgb("#444444"))[#course],
      // Push to bottom
      v(1fr),
      text(size: 12pt)[#_fmt-authors(author)],
      v(0.3em),
      text(size: 11pt, fill: rgb("#888888"))[#date],
      v(1.8em),
      image("IMADA_en.png", width: 15em),
      v(1cm),
    )
  )
  pagebreak()
  if outline { std.outline(depth: outline-depth); pagebreak() }
  base-style(body)
}

#let assignment(
  title: default-title,
  author: default-author,
  course: default-course,
  date: default-date,
  outline: true,
  outline-depth: none,
  ..args,
) = {
  let body = args.pos().at(0, default: [])
  set page(paper: "us-letter", margin: (left: 3cm, right: 3cm, top: 3cm, bottom: 3cm))
  align(center,
    stack(
      spacing: 0pt,
      v(1.2cm),
      // Amber top bar + label
      line(length: 100%, stroke: 3pt + rgb("#b7410e")),
      v(1.2em),
      text(size: 9.5pt, fill: rgb("#621e00"), tracking: 2.5pt, weight: "bold")[ASSIGNMENTS],
      v(2.5cm),
      // Title
      text(size: 30pt, weight: "bold")[#title],
      v(1.3em),
      line(length: 28%, stroke: 0.5pt + rgb("#bbbbbb")),
      v(0.7em),
      text(size: 14pt, fill: rgb("#444444"))[#course],
      // Push to bottom
      v(1fr),
      text(size: 12pt)[#_fmt-authors(author)],
      v(1.8em),
      text(size: 11pt, fill: rgb("#888888"))[#date],
      v(1.8em),
      image("IMADA_en.png", width: 15em),
      v(1cm),
    )
  )
  pagebreak()
  if outline { std.outline(depth: outline-depth); pagebreak() }
  base-style(body)
}

#let project(
  title: default-title,
  subtitle: none,
  author: default-author,
  course: default-course,
  date: default-date,
  group: none,
  supervisor: none,
  university: "University of Southern Denmark",
  abstract: none,
  keywords: none,
  outline: true,
  outline-depth: none,
  ..args,
) = {
  let body = args.pos().at(0, default: [])
  set page(paper: "us-letter", margin: (left: 3cm, right: 3cm, top: 3cm, bottom: 3cm))
  align(center,
    stack(
      spacing: 0pt,
      // Top: university name
      v(1.5cm),
      text(size: 13pt, fill: rgb("#555555"))[#university],
      v(0.6em),
      line(length: 60%, stroke: 0.5pt + rgb("#aaaaaa")),
      v(3cm),

      // Title block
      text(size: 28pt, weight: "bold")[#title],
      if subtitle != none {
        stack(
          v(1.5em),
          text(size: 15pt, fill: rgb("#444444"), style: "italic")[#subtitle],
        )
      },
      v(1em),
      line(length: 40%, stroke: 0.5pt + rgb("#aaaaaa")),
      v(1.3em),
      text(size: 14pt, fill: rgb("#333333"))[#course],

      // Fill remaining space
      v(1fr),

      // Authors — 3/2 centered grid layout
      {
        let author-arr = if type(author) == str {
          ((name: author),)
        } else if type(author) == array and author.len() > 0 and type(author.at(0)) == str {
          author.map(n => (name: n))
        } else if type(author) == array {
          author
        } else { ((name: str(author)),) }

        let render-author(a) = align(center, stack(
          spacing: 0.25em,
          text(weight: "bold", size: 11pt)[#a.at("name", default: "")],
          if a.at("email", default: "") != "" {
            text(size: 8.5pt, fill: rgb("#4a90d9"))[#a.at("email", default: "")]
          },
        ))

        let per-row = 3
        let row-starts = range(0, author-arr.len(), step: per-row)
        stack(spacing: 1.5em,
          ..row-starts.map(i => {
            let row = author-arr.slice(i, calc.min(i + per-row, author-arr.len()))
            align(center,
              box(width: (100% * row.len() / per-row),
                grid(
                  columns: (1fr,) * row.len(),
                  column-gutter: 2em,
                  ..row.map(render-author),
                )
              )
            )
          })
        )
      },
      v(1.5em),

      // Metadata box (group / supervisor / date)
      block(
        width: 60%,
        stroke: (top: 0.5pt + rgb("#aaaaaa"), bottom: 0.5pt + rgb("#aaaaaa")),
        inset: (top: 1em, bottom: 1em),
        align(left,
          stack(
            spacing: 0.5em,
            if group != none {
              grid(
                columns: (4cm, 1fr),
                text(fill: rgb("#777777"))[*Group:*],
                text()[#group],
              )
            },
            if supervisor != none {
              grid(
                columns: (4cm, 1fr),
                text(fill: rgb("#777777"))[*Supervisor:*],
                text()[#supervisor],
              )
            },
            grid(
              columns: (4cm, 1fr),
              text(fill: rgb("#777777"))[*Date:*],
              text()[#date],
            ),
          )
        )
      ),

      // Abstract block
      if abstract != none {
        stack(
          spacing: 0pt,
          v(1.5em),
          block(
            width: 80%,
            stroke: none,
            inset: (top: 0em, bottom: 0em),
            align(left, stack(
              spacing: 0.5em,
              text(weight: "bold", size: 10pt, fill: rgb("#333333"))[Abstract],
              line(length: 100%, stroke: 0.4pt + rgb("#cccccc")),
              v(0.3em),
              text(size: 9.5pt, fill: rgb("#444444"))[#abstract],
            ))
          ),
        )
      },

      // Keywords
      if keywords != none {
        stack(
          spacing: 0pt,
          v(0.8em),
          block(
            width: 80%,
            inset: 0pt,
            align(left,
              text(size: 9.5pt)[
                #text(weight: "bold", fill: rgb("#333333"))[Keywords: ]
                #text(fill: rgb("#555555"))[
                  #if type(keywords) == array { keywords.join(", ") } else { keywords }
                ]
              ]
            )
          ),
        )
      },

      v(1.8em),
      image("IMADA_en.png", width: 15em),
      v(1cm),
    )
  )
  pagebreak()
  if outline { std.outline(depth: outline-depth); pagebreak() }
  base-style(body)
}

#let exam(
  title: default-title,
  subtitle: none,
  author: default-author,
  course: default-course,
  date: default-date,
  student-id: none,
  username: none,
  student-number: none,
  duration: none,
  allowed-aids: none,
  university: "University of Southern Denmark",
  outline: true,
  outline-depth: none,
  ..args,
) = {
  let body = args.pos().at(0, default: [])
  let author-name = if type(author) == str { author }
    else if type(author) == array and author.len() > 0 {
      if type(author.at(0)) == str { author.at(0) }
      else { author.at(0).at("name", default: "") }
    } else { "" }
  set page(
    paper: "us-letter",
    margin: (left: 3cm, right: 3cm, top: 3cm, bottom: 3cm),
    header: if username != none or student-number != none {
      set text(size: 9pt, fill: rgb("#555555"))
      grid(
        columns: (1fr, 1fr, 1fr),
        align(left)[#author-name],
        align(center)[#if username != none { username }],
        align(right)[#if student-number != none { student-number }],
      )
    },
  )
  align(center,
    stack(
      spacing: 0pt,
      // Top: university name
      v(1.5cm),
      text(size: 13pt, fill: rgb("#555555"))[#university],
      v(0.6em),
      line(length: 60%, stroke: 0.5pt + rgb("#aaaaaa")),
      v(0.5cm),

      // Green label
      text(size: 9.5pt, fill: rgb("#1a6b3c"), tracking: 2.5pt, weight: "bold")[EXAM],
      v(4.5cm),

      // Title block
      text(size: 28pt, weight: "bold")[#title],
      if subtitle != none {
        stack(
          v(1.5em),
          text(size: 15pt, fill: rgb("#444444"), style: "italic")[#subtitle],
        )
      },
      v(1em),
      line(length: 40%, stroke: 0.5pt + rgb("#aaaaaa")),
      v(1.3em),
      text(size: 14pt, fill: rgb("#333333"))[#course],

      // Fill remaining space
      v(1fr),

      // Authors
      {
        let author-arr = if type(author) == str {
          ((name: author),)
        } else if type(author) == array and author.len() > 0 and type(author.at(0)) == str {
          author.map(n => (name: n))
        } else if type(author) == array {
          author
        } else { ((name: str(author)),) }

        let render-author(a) = align(center, stack(
          spacing: 0.25em,
          text(weight: "bold", size: 11pt)[#a.at("name", default: "")],
          if a.at("id", default: "") != "" {
            text(size: 9pt, fill: rgb("#555555"))[#a.at("id", default: "")]
          },
        ))

        let per-row = 3
        let row-starts = range(0, author-arr.len(), step: per-row)
        stack(spacing: 1.5em,
          ..row-starts.map(i => {
            let row = author-arr.slice(i, calc.min(i + per-row, author-arr.len()))
            align(center,
              box(width: (100% * row.len() / per-row),
                grid(
                  columns: (1fr,) * row.len(),
                  column-gutter: 2em,
                  ..row.map(render-author),
                )
              )
            )
          })
        )
      },
      v(1.5em),

      // Metadata box
      block(
        width: 60%,
        stroke: (top: 0.5pt + rgb("#aaaaaa"), bottom: 0.5pt + rgb("#aaaaaa")),
        inset: (top: 1em, bottom: 1em),
        align(left,
          stack(
            spacing: 0.5em,
            if duration != none {
              grid(
                columns: (4cm, 1fr),
                text(fill: rgb("#777777"))[*Duration:*],
                text()[#duration],
              )
            },
            if allowed-aids != none {
              grid(
                columns: (4cm, 1fr),
                text(fill: rgb("#777777"))[*Allowed aids:*],
                text()[#allowed-aids],
              )
            },
            grid(
              columns: (4cm, 1fr),
              text(fill: rgb("#777777"))[*Date:*],
              text()[#date],
            ),
          )
        )
      ),
      v(1.8em),
      image("IMADA_en.png", width: 15em),
      v(1cm),
    )
  )
  pagebreak()
  if outline { std.outline(depth: outline-depth); pagebreak() }
  base-style(body)
}



#let chi(
  title: default-title,
  authors: (),       // string or array of dicts: (name:, institution:, city:, country:, email:)
  abstract: [],
  keywords: (),
  ccs: none,
  date: default-date,
  outline: false,
  ..args,
) = {
  let body = args.pos().at(0, default: [])
  set page(paper: "us-letter", margin: (x: 1.9cm, y: 2.3cm))
  set text(size: 9.5pt)

  // Title
  v(0.5cm)
  align(center, text(size: 18pt, weight: "bold")[#title])
  v(1.5em)

  // Authors — accepts either a string or an array of dicts
  let authors-arr = if type(authors) == str {
    ((name: authors),)
  } else if authors.len() > 0 and type(authors.at(0)) == str {
    authors.map(n => (name: n))
  } else {
    authors
  }
  if authors-arr.len() > 0 {
    let render-author(a) = align(center, stack(
      spacing: 0.3em,
      text(weight: "bold", size: 10.5pt)[#a.at("name", default: "")],
      if a.at("institution", default: "") != "" { text(size: 9pt)[#a.at("institution", default: "")] },
      if a.at("city", default: "") != "" or "country" in a {
        text(size: 9pt)[#a.at("city", default: "")#if "country" in a [, #a.country]]
      },
      if a.at("email", default: "") != "" {
        text(size: 9pt, fill: rgb("#0055aa"))[#a.at("email", default: "")]
      },
    ))

    // CHI layout: up to 3 per row, partial last row centered
    let n = authors-arr.len()
    let row-starts = range(0, n, step: 3)
    for i in row-starts {
      let row = authors-arr.slice(i, calc.min(i + 3, n))
      align(center,
        box(width: (100% * row.len() / 3),
          grid(columns: (1fr,) * row.len(), column-gutter: 2em,
            ..row.map(render-author))
        )
      )
      if i + 3 < n { v(1.5em) }
    }

    v(1.8em)
    align(center, text(size: 9pt, fill: rgb("#888888"))[#date])
    v(1em)
  }

  // Abstract + CCS + Keywords — only rendered if any are provided
  let has-meta = abstract != [] or keywords != () or ccs != none
  if has-meta {
    line(length: 100%, stroke: 0.5pt + rgb("#888888"))
    v(1em)
    columns(2, gutter: 1.5em, [
      #if abstract != [] {
        text(weight: "bold", size: 8.5pt, tracking: 0.8pt)[ABSTRACT]
        v(0.4em)
        abstract
      }
      #if ccs != none {
        v(0.8em)
        text(weight: "bold", size: 8.5pt, tracking: 0.8pt)[CCS CONCEPTS]
        v(0.4em)
        ccs
      }
      #if keywords != () {
        v(0.8em)
        text(weight: "bold", size: 8.5pt, tracking: 0.8pt)[KEYWORDS]
        v(0.4em)
        keywords.join("; ")
      }
    ])
  }

  line(length: 100%, stroke: 0.5pt + rgb("#888888"))
  v(1em)
  if outline { pagebreak(); std.outline(); pagebreak() }
  base-style(body)
}

// venn — set diagram: elements placed symmetrically inside P, ovals fitted to subsets
//
// Elements are arranged on a ring inside P. Each subset gets a rotated bounding
// oval oriented along its principal axis, so {p,r} with p at top and r at right
// produces a diagonal oval — not a circle. Shared elements fall inside all ovals
// that contain them.
//
// Example:
//   #venn(
//     domain: $P$,
//     ($A$, ("p", "q")),
//     ($B$, ("q", "r", "s")),
//   )
#let venn(domain: none, outside: (), scale: 2cm, universe-fill: none, ..args) = {
  let sets = args.pos().map(s => {
    let mem = s.at(1)
    let col = s.at(2, default: auto)  // auto = use palette, none = unfilled, color = that color
    (s.at(0), if type(mem) == str { (mem,) } else { mem }, col)
  })
  let R        = 2.8
  let er       = R * 0.52
  let pad      = 0.58   // along major axis (elements sit at tips, need room)
  let pad-perp = 0.32   // perpendicular (no elements here, stay tight)

  let palette = (
    red.transparentize(75%), blue.transparentize(75%),
    green.transparentize(75%), orange.transparentize(75%),
    purple.transparentize(75%), teal.transparentize(75%),
  )

  // Collect all unique elements and place on ring
  let all-elems = ()
  for s in sets { for m in s.at(1) { if m not in all-elems { all-elems.push(m) } } }
  for el in outside { if el not in all-elems { all-elems.push(el) } }
  let ne = all-elems.len()

  let epos = (:)
  for (j, el) in all-elems.enumerate() {
    let ang = 90deg - 360deg / ne * j
    epos.insert(el, (er * calc.cos(ang), er * calc.sin(ang)))
  }

  // Rotated bounding oval oriented along the principal axis of element positions
  let oval-for(members) = {
    let pts = members.map(el => epos.at(el))
    let n   = pts.len()
    let cx  = pts.map(p => p.at(0)).sum() / n
    let cy  = pts.map(p => p.at(1)).sum() / n
    if n == 1 { return (cx: cx, cy: cy, rx: pad, ry: pad-perp + 0.1, ang: 0deg) }
    let rel = pts.map(p => (p.at(0) - cx, p.at(1) - cy))
    // Principal axis = direction of farthest pair of elements
    let ax = 1.0; let ay = 0.0; let best = 0.0
    for i in range(n) {
      for j in range(i + 1, n) {
        let dx = rel.at(j).at(0) - rel.at(i).at(0)
        let dy = rel.at(j).at(1) - rel.at(i).at(1)
        let d2 = dx * dx + dy * dy
        if d2 > best { best = d2; ax = dx; ay = dy }
      }
    }
    let alen = calc.sqrt(ax * ax + ay * ay)
    let ux = ax / alen; let uy = ay / alen
    let along = rel.map(p =>  p.at(0) * ux + p.at(1) * uy)
    let perp  = rel.map(p => -p.at(0) * uy + p.at(1) * ux)
    let rx-val = along.map(calc.abs).fold(0.0, calc.max) + pad
    let ry-val = calc.max(perp.map(calc.abs).fold(0.0, calc.max) + pad-perp, rx-val * 0.38)
    (cx:  cx, cy: cy,
     rx:  rx-val,
     ry:  ry-val,
     ang: calc.atan2(ax, ay))
  }

  // Pre-compute all ovals so labels can avoid each other
  let ovs = sets.map(s => oval-for(s.at(1)))

  canvas(length: scale, {
    import draw: *
    circle((0, 0), radius: R, stroke: black + 1.2pt, fill: universe-fill)
    if domain != none { content((-R * 0.8, R * 0.88), domain) }

    // Ovals behind elements
    for (i, s) in sets.enumerate() {
      let ov  = ovs.at(i)
      let col = if s.at(2) == auto { palette.at(calc.rem(i, palette.len())) } else { s.at(2) }
      group({
        translate((ov.cx, ov.cy))
        rotate(ov.ang)
        circle((0, 0), radius: (ov.rx, ov.ry), fill: col, stroke: black + 0.8pt)
      })
      // Pick tip with most clearance from other ovals' centers
      let ux-v = calc.cos(ov.ang)
      let uy-v = calc.sin(ov.ang)
      let tip1 = (ov.cx + ov.rx * ux-v, ov.cy + ov.rx * uy-v)
      let tip2 = (ov.cx - ov.rx * ux-v, ov.cy - ov.rx * uy-v)
      let cl1 = 9999.0; let cl2 = 9999.0
      for (j, ov2) in ovs.enumerate() {
        if j != i {
          let dx1 = tip1.at(0) - ov2.cx; let dy1 = tip1.at(1) - ov2.cy
          let dx2 = tip2.at(0) - ov2.cx; let dy2 = tip2.at(1) - ov2.cy
          let d1 = calc.sqrt(dx1 * dx1 + dy1 * dy1)
          let d2 = calc.sqrt(dx2 * dx2 + dy2 * dy2)
          if d1 < cl1 { cl1 = d1 }
          if d2 < cl2 { cl2 = d2 }
        }
      }
      let (tx, ty) = if cl1 >= cl2 { tip1 } else { tip2 }
      let td = calc.sqrt(tx * tx + ty * ty)
      let lpos = if td > 0.05 {
        let ld = calc.min(td + 0.45, R - 0.1)
        (tx / td * ld, ty / td * ld)
      } else { (0, ov.ry + 0.5) }
      content(lpos, s.at(0))
    }

    for (el, pos) in epos {
      content(pos, text(style: "italic", el))
    }
  })
}

/*
=============================================================
TEMPLATE CHEATSHEET — copy the block you need into a new file
=============================================================

── NOTE ──────────────────────────────────────────────────────
#import "../../temp.typ": *
#show: note.with(
  title:         "Lecture Notes",
  course:        "DM000 — Course Name",
  author:        "Simon Holm",
  date:          "February 2026",
  outline:       true,          // set false to skip TOC
  outline-depth: 2,             // none = unlimited depth
)

= First Section
Content goes here.

── EXERCISE ──────────────────────────────────────────────────
#import "../../temp.typ": *
#show: exercise.with(
  title:         "Exercises 1",
  course:        "DM000 — Course Name",
  author:        "Simon Holm",
  date:          "February 2026",
  outline:       true,
  outline-depth: 2,
)

= Exercise 1
// Content goes here.

── ASSIGNMENT ────────────────────────────────────────────────
#import "../../temp.typ": *
#show: assignment.with(
  title:         "Assignment 1",
  course:        "DM000 — Course Name",
  author:        "Simon Holm",
  date:          "February 2026",
  outline:       true,
  outline-depth: 2,
)

= Problem 1
// Content goes here.

── EXAM ──────────────────────────────────────────────────────
#import "../../temp.typ": *
#show: exam.with(
  title:         "Written Exam",
  subtitle:      "Re-exam",                    // optional
  course:        "DM000 — Course Name",
  author:        "Simon Holm",
  date:          "June 2026",
  student-id:    "sihol24",                    // optional
  username:      "sihol24",                    // optional — shown in page header
  student-number: "215751682",                 // optional — shown in page header
  duration:      "4 hours",                    // optional
  allowed-aids:  "All written materials",      // optional
  university:    "University of Southern Denmark",
  outline:       false,
)

= Problem 1
// Content goes here.

── EXAM (group / multiple students) ─────────────────────────
#import "../../temp.typ": *
#show: exam.with(
  title:        "Written Exam",
  course:       "DM000 — Course Name",
  author: (
    (name: "Simon Holm", id: "sihol24"),
    (name: "Firstname Lastname",   id: "jado42"),
  ),
  date:         "June 2026",
  duration:     "4 hours",
  allowed-aids: "None",
)

= Problem 1
// Content goes here.

── PROJECT ───────────────────────────────────────────────────
#import "../../temp.typ": *
#show: project.with(
  title:         "Project Title",
  subtitle:      "Optional subtitle",          // optional
  course:        "DM000 — Course Name",
  author:        "Simon Holm",                 // or array of dicts below
  date:          "February 2026",
  group:         "Group 4",                    // optional
  supervisor:    "Prof. Firstname Lastname",             // optional
  university:    "University of Southern Denmark",
  outline:       true,
  outline-depth: 2,
)

= Introduction
// Content goes here.

── PROJECT (multiple authors with email) ─────────────────────
#import "../../temp.typ": *
#show: project.with(
  title:  "Project Title",
  course: "DM000 — Course Name",
  author: (
    (name: "Simon Holm", email: "sihol24@student.sdu.dk"),
    (name: "Firstname Lastname",   email: "jado@student.sdu.dk"),
  ),
  date:       "February 2026",
  group:      "Group 4",
  supervisor: "Prof. Jane Doe",
)

= Introduction
// Content goes here.

── CHI PAPER ─────────────────────────────────────────────────
#import "../../temp.typ": *
#show: chi.with(
  title: "Paper Title",
  authors: (
    (name: "Simon Holm", institution: "University of Southern Denmark", city: "Odense", country: "Denmark", email: "sihol24@student.sdu.dk"),
    (name: "Author Two", institution: "University of Southern Denmark", city: "Odense", country: "Denmark", email: "two@student.sdu.dk"),
  ),
  abstract: [Your abstract text here.],
  keywords: ("keyword one", "keyword two", "keyword three"),
  ccs:      [\u{2192} Human-centered computing \u{2192} HCI theory, concepts and models], // optional
  date:     "March 2026",
  outline:  false,
)
#set page(columns: 2)

= Introduction
// Content goes here.

── VENN DIAGRAM (3-set) ──────────────────────────────────────
// Domain P = {p,q,r}, denotations: [|p|]={p,r}, [|q|]={p,q,r}, [|r|]={r}
#venn3(
  label-a: $[|p|]$, label-b: $[|q|]$, label-c: $[|r|]$,
  ab:      ($p$,),    // p ∈ [|p|] ∩ [|q|], not [|r|]
  abc:     ($r$,),    // r ∈ [|p|] ∩ [|q|] ∩ [|r|]
  only-b:  ($q$,),    // q ∈ [|q|] only
)

── MAPPING DIAGRAM (inline, no import needed) ────────────────
#mapdiag(
  title:        $f: A -> B$,           // optional label above diagram
  a:            $A$,                   // left set label  (default $A$)
  b:            $B$,                   // right set label (default $B$)
  a-elems:      ($1$, $2$, $3$),
  b-elems:      ($a$, $b$, $c$),
  arrow-color:  black,                 // default arrow colour
  arrows: (
    (0, 0),                            // plain arrow
    (1, 2, red),                       // coloured arrow
    (2, 1, blue, $g$),                 // coloured + label
  ),
)
*/
