#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 10pt, fill: black)
#set text(fill: white)

// 784 -> 128 -> 64 -> 10

#let r = 0.5          // neuron radius
#let dy = 1.5         // vertical spacing
#let layer-x = (0, 8, 16, 24)

// Label a_k^(l)
#let lab(k, l) = $a_#k^(#l)$

// Compute the (pos, label) pairs for one "big" layer of n neurons:
// 7 neurons on top, a vertical-dots gap, then 8 neurons on the bottom.
#let big-layer(x, l, n) = {
  let nodes = ()
  for i in range(1, 8) {
    nodes.push(((x, -i * dy), lab(i, l)))
  }
  for i in range(8, 16) {
    nodes.push(((x, -i * dy - dy), lab(n - 15 + i, l)))
  }
  nodes
}

// Compute the (pos, label) pairs for the small output layer of n neurons.
#let out-layer(x, l, n) = {
  range(1, n + 1).map(i => ((x, -i * dy - 3 * dy), lab(i, l)))
}

#let I = big-layer(layer-x.at(0), 0, 784)
#let H1 = big-layer(layer-x.at(1), 1, 128)
#let H2 = big-layer(layer-x.at(2), 2, 64)
#let O = out-layer(layer-x.at(3), 3, 10)

#cetz.canvas({
  import cetz.draw: *

  // Fully-connected edges first, so neurons sit on top of the lines.
  let connect(a, b) = {
    for (p, _) in a {
      for (q, _) in b {
        line(p, q, stroke: white)
      }
    }
  }
  connect(I, H1)
  connect(H1, H2)
  connect(H2, O)

  // Neurons (filled black so the lines don't show through the circles).
  let draw-nodes(nodes) = {
    for (p, label) in nodes {
      circle(p, radius: r, stroke: white, fill: black)
      content(p, label)
    }
  }
  draw-nodes(I)
  draw-nodes(H1)
  draw-nodes(H2)
  draw-nodes(O)

  // Vertical dots in the gap of each big layer.
  for x in (layer-x.at(0), layer-x.at(1), layer-x.at(2)) {
    content((x, -8 * dy), text(fill: white)[$dots.v$])
  }
})
