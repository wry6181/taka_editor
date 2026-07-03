use glam::Vec3;

pub const INVALID: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
pub struct HalfEdge {
    pub twin: u32,
    pub next: u32,
    pub prev: u32,
    pub vertex: u32,
    pub face: u32,
    pub edge: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub pos: Vec3,
    pub half_edge: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Face {
    pub half_edge: u32,
}

#[derive(Clone, Default)]
pub struct HalfEdgeMesh {
    pub vertices: Vec<Vertex>,
    pub half_edges: Vec<HalfEdge>,
    pub faces: Vec<Face>,
}

impl HalfEdgeMesh {
    pub fn from_triangles(positions: &[Vec3], indices: &[u32]) -> Self {
        let mut mesh = HalfEdgeMesh {
            vertices: positions
                .iter()
                .map(|&pos| Vertex { pos, half_edge: INVALID })
                .collect(),
            half_edges: Vec::with_capacity(indices.len()),
            faces: Vec::with_capacity(indices.len() / 3),
        };

        let mut edge_map: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();

        for tri in indices.chunks_exact(3) {
            let face_idx = mesh.faces.len() as u32;
            let base = mesh.half_edges.len() as u32;

            for i in 0..3 {
                let from = tri[i];
                let to = tri[(i + 1) % 3];
                let he_idx = base + i as u32;

                mesh.half_edges.push(HalfEdge {
                    twin: INVALID,
                    next: base + ((i + 1) % 3) as u32,
                    prev: base + ((i + 2) % 3) as u32,
                    vertex: from,
                    face: face_idx,
                    edge: INVALID,
                });

                mesh.vertices[from as usize].half_edge = he_idx;

                if let Some(&twin_idx) = edge_map.get(&(to, from)) {
                    mesh.half_edges[he_idx as usize].twin = twin_idx;
                    mesh.half_edges[twin_idx as usize].twin = he_idx;
                }
                edge_map.insert((from, to), he_idx);
            }

            mesh.faces.push(Face { half_edge: base });
        }

        mesh
    }

    pub fn vertex_outgoing(&self, v: u32) -> impl Iterator<Item = u32> + '_ {
        let start = self.vertices[v as usize].half_edge;
        std::iter::successors(Some(start), move |&he| {
            let twin = self.half_edges[he as usize].twin;
            if twin == INVALID {
                return None;
            }
            let next = self.half_edges[twin as usize].next;
            (next != start).then_some(next)
        })
    }

    pub fn face_vertices(&self, f: u32) -> impl Iterator<Item = u32> + '_ {
        let start = self.faces[f as usize].half_edge;
        std::iter::successors(Some(start), move |&he| {
            let next = self.half_edges[he as usize].next;
            (next != start).then_some(next)
        })
        .map(move |he| self.half_edges[he as usize].vertex)
    }

    pub fn to_triangles(&self) -> (Vec<Vec3>, Vec<u32>) {
        let positions: Vec<Vec3> = self.vertices.iter().map(|v| v.pos).collect();
        let mut indices = Vec::with_capacity(self.faces.len() * 3);
        for f in 0..self.faces.len() as u32 {
            indices.extend(self.face_vertices(f));
        }
        (positions, indices)
    }

    /// Flip the shared edge between two triangles.
    pub fn edge_flip(&mut self, he: u32) {
        let twin = self.half_edges[he as usize].twin;
        if twin == INVALID {
            return;
        }

        let he_next = self.half_edges[he as usize].next;
        let twin_next = self.half_edges[twin as usize].next;

        let a = self.half_edges[he_next as usize].vertex;
        let b = self.half_edges[twin_next as usize].vertex;
        let face_a = self.half_edges[he as usize].face;
        let face_b = self.half_edges[twin as usize].face;

        self.half_edges[he as usize].vertex = a;
        self.half_edges[he as usize].next = he_next;
        self.half_edges[he as usize].prev = twin_next;

        self.half_edges[twin as usize].vertex = b;
        self.half_edges[twin as usize].next = twin_next;
        self.half_edges[twin as usize].prev = he_next;

        self.half_edges[he_next as usize].next = he;
        self.half_edges[he_next as usize].prev = twin;
        self.half_edges[he_next as usize].face = face_a;

        self.half_edges[twin_next as usize].next = twin;
        self.half_edges[twin_next as usize].prev = he;
        self.half_edges[twin_next as usize].face = face_b;

        self.faces[face_a as usize].half_edge = he;
        self.faces[face_b as usize].half_edge = twin;

        self.vertices[a as usize].half_edge = he;
        self.vertices[b as usize].half_edge = twin;
    }

    /// Split an edge by inserting a vertex at the midpoint.
    /// Returns the new vertex index.
    pub fn edge_split(&mut self, he: u32) -> u32 {
        let twin = self.half_edges[he as usize].twin;
        if twin == INVALID {
            return INVALID;
        }

        let v0 = self.half_edges[he as usize].vertex;
        let v1 = self.half_edges[twin as usize].vertex;
        let face_a = self.half_edges[he as usize].face;
        let face_b = self.half_edges[twin as usize].face;

        // New vertex at midpoint
        let mid = (self.vertices[v0 as usize].pos + self.vertices[v1 as usize].pos) * 0.5;
        let mid_idx = self.vertices.len() as u32;
        self.vertices.push(Vertex { pos: mid, half_edge: he });

        // Save neighbors before modifying
        let he_next = self.half_edges[he as usize].next;
        let _he_prev = self.half_edges[he as usize].prev;
        let twin_next = self.half_edges[twin as usize].next;
        let _twin_prev = self.half_edges[twin as usize].prev;

        // Create two new half-edges: he2 (mid->v1) and twin2 (v1->mid)
        let he2 = self.half_edges.len() as u32;
        self.half_edges.push(HalfEdge {
            twin: he2 + 1,
            next: he_next,
            prev: he,
            vertex: mid_idx,
            face: face_a,
            edge: INVALID,
        });

        let twin2 = self.half_edges.len() as u32;
        self.half_edges.push(HalfEdge {
            twin: he2,
            next: twin_next,
            prev: twin,
            vertex: v1,
            face: face_b,
            edge: INVALID,
        });

        // Update existing half-edges
        self.half_edges[he as usize].next = he2;
        self.half_edges[twin as usize].vertex = mid_idx;
        self.half_edges[twin as usize].next = twin2;

        self.half_edges[he_next as usize].prev = he2;
        self.half_edges[twin_next as usize].prev = twin2;

        self.vertices[mid_idx as usize].half_edge = he2;

        mid_idx
    }

    /// Collapse edge (v0,v1), merging v0 into v1.
    /// Returns the surviving vertex index.
    pub fn edge_collapse(&mut self, he: u32) -> u32 {
        let twin = self.half_edges[he as usize].twin;
        if twin == INVALID {
            return INVALID;
        }

        let v0 = self.half_edges[he as usize].vertex;
        let _v1 = self.half_edges[twin as usize].vertex;
        let face_a = self.half_edges[he as usize].face;
        let face_b = self.half_edges[twin as usize].face;

        if face_a == face_b {
            return INVALID;
        }

        // Half-edges of the two faces (CCW):
        // Face A: he_prev -> he -> he_next
        // Face B: twin_prev -> twin -> twin_next
        let he_next = self.half_edges[he as usize].next;
        let he_prev = self.half_edges[he as usize].prev;
        let twin_next = self.half_edges[twin as usize].next;
        let twin_prev = self.half_edges[twin as usize].prev;

        let _va = self.half_edges[he_next as usize].vertex;
        let _vb = self.half_edges[twin_next as usize].vertex;

        // Stitch: he_prev and twin_prev become a new edge a<->b
        // Both get face of whatever remains (reuse face_a arbitrarily)
        self.half_edges[he_prev as usize].next = twin_next;
        self.half_edges[he_prev as usize].prev = twin_prev;
        self.half_edges[twin_next as usize].prev = he_prev;
        self.half_edges[twin_prev as usize].next = he_prev;

        // Mark twin as new boundary (or just keep both — it'll be claimed below)
        self.half_edges[he_next as usize].next = he_prev;
        self.half_edges[he_next as usize].prev = twin_prev;

        // Remove the two faces
        // We need to handle swapped-index removal
        let max_face = self.faces.len() as u32 - 1;
        for &rm in &[face_a, face_b] {
            if rm == max_face {
                self.faces.swap_remove(rm as usize);
            } else {
                self.faces.swap_remove(rm as usize);
                // Update any half-edge pointing to the swapped face
                let swapped_face_idx = rm;
                for he in 0..self.half_edges.len() as u32 {
                    if self.half_edges[he as usize].face == max_face {
                        self.half_edges[he as usize].face = swapped_face_idx;
                    }
                }
            }
        }

        v0
    }
}
