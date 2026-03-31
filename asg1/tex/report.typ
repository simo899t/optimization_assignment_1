#import "/temp/temp.typ": *

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

For this we define a class in python to handle the seach space.

#code(
  ```py
  class search_space():
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.obstacles = []
        self.trajectory = []
  ```
)

=== Straight path
For a straight path we can generate a path for finding $x$

The straigt path can be easily found by

#algorithm(caption: [Straight path])[
  #pseudo[
    - *Input:* start $x_1 = (1,1)$, goal $x_n = (100,100)$, steps $s = 20$
    - *Output:* trajectory $T$
    + direction $d arrow.l x_n - x_1$
    + $T arrow.l []$
    + *for* $t in "linspace"(0, 1, s)$ *do*
      + $p arrow.l x_1 + t dot d$
      + append $p$ to $T$
    + *return* $T$
  ]
]
#pagebreak()

=== Obstacles

We can define a new class and an `add_object` function to the `search_space` class
#code(
```py
class circular_object():
    def __init__(self, center: np.array, diameter: float):
        self.center_point = center
        self.diameter = diameter

class search_space():
  # ...
  def add_obstacle(self, object: circular_object):
        self.obstacles.append(object)
```
)

This way its easy to create object within the seach space

#code(
```py
search = search_space(x0, xn)

object1 = circular_object(np.array([40,30]), 10.0)
search.add_obstacle(object1)
object2 = circular_object(np.array([50,55]), 7.0)
search.add_obstacle(object2)
object3 = circular_object(np.array([60,80]), 15.0)
search.add_obstacle(object3)
```
)

=== Visualisation
#figure(
  image("assets/image-2.png", width: 30em),
  caption: [Linear path with tree obstacles\
   `[circular_object(c=[40 30], d=20.0), circular_object(c=[50 55], d=15.0), circular_object(c=[60 80], d=30.0)]` ]
)

#pagebreak()

== Objective Function

The objective function is defined as
$ f(bold(x)) = f_L (bold(x)) + lambda dot f_S (bold(x)) + mu dot f_O (bold(x)) $
or in python
#code(
```py
class search_space():
  # ...
  def obj_func(self, x, lam, mu, alpha):
        def f(x):
            traj = x.reshape(-1, 2)
            return (self.pathlength(traj)
                    + lam * self.smoothness(traj)
                    + mu * self.avoidance(traj, alpha))

        return f(x), grad(f)(x)
```
)
=== Path length
The path length function is defined as $ f_L (bold(x)) = summ(i=1, n-1, norm(x_(i+1) - x_i)^2) $
or in python
#code(
```py
class search_space():
  # ...
  def pathlength(self):
          path = self.trajectory
          return sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path) - 1))
```
)

=== Smoothnes

The smoothnes function is defined as $ f_S (bold(x)) = summ(i=2, n-1, norm(x_(i+1) - 2x_i- x_(i-1))^2) $
or in python
#code(
```py
class search_space():
  # ...
  def smoothness(self):
        x = self.trajectory
        return sum(np.linalg.norm(x[i+1] - 2*x[i]- x[i-1]) for i in range(1,len(x) - 1))
```
)
#pagebreak()

=== Object avoidance
The obstacle avoidance function is defined as
$ f_O (bold(x)) = summ(i=1,n,phi(x_i))) $
$ phi(bold(x)_i) = exp(-alpha(d(bold(x)_i)^2 - r^2)) $

#code(
```py
class search_space():
  # ...
  def avoidance(self, x, alpha):
        penalty = 0
        for xi in x:
            penalty += sum(anp.exp(-alpha * (
                anp.linalg.norm(xi - obs.center_point)**2 - obs.radius**2))
                    for obs in self.obstacles
            )
        return penalty
```
)

#figure(
  image("assets/image.png", width: 30em),
  caption: [der sker en ting med]
)

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
