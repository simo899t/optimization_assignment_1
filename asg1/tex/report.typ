#import "@preview/lovelace:0.3.0": pseudocode-list
#let alg = pseudocode-list

// ─── Settings ────────────────────────────────────────────────────────────────
#let courseid   = "AI505"
#let coursename = "Optimization"
#let term       = "Spring 2025"
#let dept       = "Department of Mathematics and Computer Science"
#let university = "University of Southern Denmark, Odense"
#let authors    = "Simon Holm, Johannes Rothe"
#let assnr      = "1"

// ─── Page layout ─────────────────────────────────────────────────────────────
#set page(
  paper: "a4",
  margin: 2.7cm,
  header: context {
    set text(size: 10pt)
    if counter(page).get().first() == 1 {
      grid(
        columns: (1fr, auto),
        align: (left + horizon, right + horizon),
        [#dept \ #university],
        [#datetime.today().display("[day] [month repr:long] [year]") \ #authors],
      )
      v(-6pt)
      line(length: 100%, stroke: 0.4pt)
    } else {
      align(right)[#authors]
      v(-6pt)
      line(length: 100%, stroke: 0.4pt)
    }
  },
  footer: context {
    line(length: 100%, stroke: 0.4pt)
    v(-6pt)
    align(center)[
      #set text(size: 10pt)
      Page #counter(page).display() of #counter(page).final().first()
    ]
  },
)

// ─── Typography ───────────────────────────────────────────────────────────────
#set text(font: "New Computer Modern", size: 11pt)
#set par(leading: 0.65em, spacing: 2em, first-line-indent: 0pt)

// Number equations
#set math.equation(numbering: "(1)")

// Section numbering
#set heading(numbering: "1.1")

// ─── Algorithm block helper ───────────────────────────────────────────────────
#let algorithm(caption: none, body) = {
  set text(size: 10pt)
  block(
    width: 100%,
    stroke: (top: 1pt, bottom: 1pt),
    inset: (x: 8pt, y: 6pt),
  )[
    #if caption != none [
      *Algorithm:* #caption \
      #line(length: 100%, stroke: 0.4pt)
      #v(-16pt)
    ]
    #body
  ]
}


// ─── Code listing helper ──────────────────────────────────────────────────────
#show raw.where(block: true): it => {
  block(
    width: 100%,
    stroke: (top: 0.6pt, bottom: 0.6pt),
    inset: (x: 8pt, y: 6pt),
    fill: luma(248),
  )[#it]
}

// ─── Title ────────────────────────────────────────────────────────────────────
#v(-4pt)
#align(left)[
  #text(size: 11pt)[#courseid -- #coursename]
  #v(0.2cm)
  #text(size: 14pt, weight: "bold")[
    Answers to Obligatory Assignment #assnr, #term
  ]
  #v(0.4em)
  #line(length: 100%, stroke: 0.8pt)
]

#v(1em)

// ─── Template instructions (remove for submission) ────────────────────────────
Always insert a `#pagebreak()` after every Task.
Remove the content of this page.

#v(1em)

#line(length: 100%, stroke: (dash: "dashed", thickness: 0.4pt))
*Template to write models (see source)*

$
  max quad & sum_(j=1)^n c_j x_j                              & & #h(0em) \
  "s.t." quad & sum_(j=1)^n a_(i j) x_j >= b_i,  quad & i = 1, dots, m \
           & x_j >= 0,                             quad & j = 1, dots, n
$

#line(length: 100%, stroke: (dash: "dashed", thickness: 0.4pt))
*Template for algorithm pseudocode (see source):*

#v(0.5em)
#algorithm(caption: [How to write algorithms])[
  #pseudocode-list[
    - *Data:* this text
    - *Result:* how to write algorithm with Typst
    + initialization
    + *while* not at end of this document *do*
      + read current
      + *if* understand *then*
        + go to next section
        + current section becomes this one
      + *else*
        + go back to the beginning of current section
  ]
]

#v(0.5em)
#line(length: 100%, stroke: (dash: "dashed", thickness: 0.4pt))
*Template for source code inclusion:*

```python
import numpy as np
```

#line(length: 100%, stroke: (dash: "dashed", thickness: 0.4pt))



// ─── Task 1 ───────────────────────────────────────────────────────────────────
#pagebreak()
= Task 1
== Problem Setup
=== Initial problem
A robot must move from a given start position to a given goal position in a 2D environment while avoiding obstacles and producing a smooth trajectory.

Since the trajectory is defined as
$ {x_1,x_2,dots,x_n} in RR^2 $

Lets fix 2 points, that is the starting point $x_1$ and the goal point $x_n$

$ x_1 = vec(1,1), quad x = vec(100,100) $

=== Straight path
For a straight path we can generate a path for finding $x$


== Objective Function
== Optimization Algorithms
== Comparison




// ─── Task 2 ───────────────────────────────────────────────────────────────────
#pagebreak()
= Task 2



// ─── Appendix ─────────────────────────────────────────────────────────────────
#pagebreak()
#set heading(numbering: "A")
#counter(heading).update(0)
= Appendix <appendix>

Here we should talk about AI use in the project.

_In your submission you have to add an Appendix where you declare to which extent you
have used AI tools.
The Appendix must be present also if you did not use AI and hence declare so._

_If you do not declare but the teachers have suspects that you used AI tools,
you will be reported for exam cheating._

_If you use AI, the teachers will be allowed to take this in consideration and to lower
consequently the final grade if they suspect that the learning goals have not been achieved._

_In declaring the use you can share the conversation you had with the tools and provide
the link or copying it in the Appendix.
The Appendix is not subjected to page limits._
