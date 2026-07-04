use glam::{Mat4, Vec2, Vec3};

use super::half_edge::HalfEdgeMesh;
use super::ray::Ray;

pub fn rdp_simplify(points: &[Vec2], epsilon: f32) -> Vec<usize> {
    if points.len() <= 2 {
        return (0..points.len()).collect();
    }
    let mut kept = Vec::new();
    rdp_recursive(points, epsilon, 0, points.len() - 1, &mut kept);
    kept.sort_unstable();
    kept.dedup();
    kept
}

fn rdp_recursive(points: &[Vec2], epsilon: f32, first: usize, last: usize, kept: &mut Vec<usize>) {
    if first + 1 >= last {
        kept.push(first);
        kept.push(last);
        return;
    }
    let a = points[first];
    let b = points[last];
    let ab = b - a;
    let len_sq = ab.length_squared();
    let (mut max_dist, mut max_idx) = (0.0f32, first);
    for i in (first + 1)..last {
        let dist = if len_sq < 1e-12 {
            (points[i] - a).length()
        } else {
            let t = ((points[i] - a).dot(ab) / len_sq).clamp(0.0, 1.0);
            (points[i] - (a + ab * t)).length()
        };
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }
    if max_dist > epsilon {
        rdp_recursive(points, epsilon, first, max_idx, kept);
        rdp_recursive(points, epsilon, max_idx, last, kept);
    } else {
        kept.push(first);
        kept.push(last);
    }
}

#[derive(Clone)]
pub struct Remesh {
    pub mesh: HalfEdgeMesh,
    pub selected: Option<usize>,
}

impl Remesh {
    pub fn new() -> Self {
        Self {
            mesh: HalfEdgeMesh::default(),
            selected: None,
        }
    }

    pub fn add_point(&mut self, pos: Vec3) -> usize {
        let idx = self.mesh.vertices.len();
        self.mesh.vertices.push(super::half_edge::Vertex {
            pos,
            half_edge: super::half_edge::INVALID,
        });

        if self.mesh.vertices.len() >= 3 {
            let (tri_positions, tri_indices) = self.mesh.to_triangles();
            let tri_pairs: Vec<(u32, u32, u32)> = tri_indices.chunks_exact(3).map(|c| (c[0], c[1], c[2])).collect();
            let mut found_inside = None;
            for (i, &(a, b, c)) in tri_pairs.iter().enumerate() {
                if point_in_triangle(pos, tri_positions[a as usize], tri_positions[b as usize], tri_positions[c as usize]) {
                    found_inside = Some(i);
                    break;
                }
            }

            let mut all_indices = tri_indices.clone();
            let vert_count = self.mesh.vertices.len();
            let mut positions: Vec<Vec3> = (0..vert_count).map(|i| self.mesh.vertices[i].pos).collect();

            if let Some(tri_idx) = found_inside {
                let (a, b, c) = tri_pairs[tri_idx];
                let tri_start = tri_idx * 3;
                all_indices.remove(tri_start);
                all_indices.remove(tri_start);
                all_indices.remove(tri_start);
                all_indices.extend_from_slice(&[a, b, idx as u32]);
                all_indices.extend_from_slice(&[b, c, idx as u32]);
                all_indices.extend_from_slice(&[c, a, idx as u32]);
            } else {
                let (i1, i2) = nearest_two(&positions, pos);
                all_indices.extend_from_slice(&[i1, i2, idx as u32]);
            }

            self.mesh = HalfEdgeMesh::from_triangles(&positions, &all_indices);
        }

        idx
    }

    pub fn add_points_batch(&mut self, positions: &[Vec3]) {
        for &p in positions {
            self.add_point(p);
        }
    }

    pub fn nearest_point(&self, pos: Vec3, threshold: f32) -> Option<usize> {
        self.mesh.vertices
            .iter()
            .enumerate()
            .map(|(i, v)| (i, (v.pos - pos).length_squared()))
            .filter(|&(_, d)| d < threshold * threshold)
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }

    pub fn select_near(&mut self, pos: Vec3) {
        self.selected = self.nearest_point(pos, 0.3);
    }

    pub fn clear_selection(&mut self) {
        self.selected = None;
    }

    pub fn gizmo_axis_hit_screen(&self, ndc_x: f32, ndc_y: f32, view: Mat4, proj: Mat4, threshold: f32) -> Option<usize> {
        let idx = self.selected?;
        let pos = self.mesh.vertices[idx].pos;
        let axis_len = 1.5;

        for axis in 0..3 {
            let dir = match axis {
                0 => Vec3::X,
                1 => Vec3::Y,
                _ => Vec3::Z,
            };
            let start_ndc = project_to_ndc(pos, view, proj);
            let end_ndc = project_to_ndc(pos + dir * axis_len, view, proj);
            let dist = ndc_segment_distance(ndc_x, ndc_y, start_ndc, end_ndc);
            if dist < threshold {
                return Some(axis);
            }
        }
        None
    }

    pub fn drag_selected(&mut self, axis: usize, ray: &Ray) {
        let Some(idx) = self.selected else { return };
        let pos = self.mesh.vertices[idx].pos;
        let axis_dir = match axis {
            0 => Vec3::X,
            1 => Vec3::Y,
            _ => Vec3::Z,
        };

        let plane_normal = axis_dir;
        let denom = plane_normal.dot(ray.direction);
        if denom.abs() < 1e-8 {
            return;
        }
        let t = plane_normal.dot(pos - ray.origin) / denom;
        if t < 0.0 {
            return;
        }
        let hit = ray.origin + ray.direction * t;
        let s = axis_dir.dot(hit - pos);
        self.mesh.vertices[idx].pos = pos + axis_dir * s;
    }

    pub fn hit_point_screen(
        &self,
        ndc_x: f32,
        ndc_y: f32,
        view: Mat4,
        proj: Mat4,
        threshold: f32,
    ) -> Option<usize> {
        for (i, v) in self.mesh.vertices.iter().enumerate() {
            let clip = proj * view * v.pos.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            let dist = ((ndc.x - ndc_x).powi(2) + (ndc.y - ndc_y).powi(2)).sqrt();
            if dist < threshold {
                return Some(i);
            }
        }
        None
    }

    pub fn set_point_position(&mut self, idx: usize, pos: Vec3) {
        if idx < self.mesh.vertices.len() {
            self.mesh.vertices[idx].pos = pos;
        }
    }

    pub fn point_count(&self) -> usize {
        self.mesh.vertices.len()
    }

    pub fn edge_flip(&mut self, he: u32) {
        self.mesh.edge_flip(he);
    }

    pub fn edge_split(&mut self, he: u32) -> u32 {
        self.mesh.edge_split(he)
    }

    pub fn edge_collapse(&mut self, he: u32) -> u32 {
        self.mesh.edge_collapse(he)
    }

    pub fn fill_vertices(&self) -> Vec<[f32; 3]> {
        let (_, indices) = self.mesh.to_triangles();
        let mut verts = Vec::with_capacity(indices.len());
        for &vi in &indices {
            verts.push(self.mesh.vertices[vi as usize].pos.to_array());
        }
        verts
    }

    pub fn wireframe_lines(&self) -> Vec<(Vec3, Vec3, Vec3)> {
        let mut lines = Vec::new();
        let color = Vec3::new(0.2, 1.0, 0.2);
        for f in 0..self.mesh.faces.len() as u32 {
            let verts: Vec<u32> = self.mesh.face_vertices(f).collect();
            if verts.len() == 3 {
                let pa = self.mesh.vertices[verts[0] as usize].pos;
                let pb = self.mesh.vertices[verts[1] as usize].pos;
                let pc = self.mesh.vertices[verts[2] as usize].pos;
                lines.push((pa, pb, color));
                lines.push((pb, pc, color));
                lines.push((pc, pa, color));
            }
        }
        lines
    }

    pub fn faces_empty(&self) -> bool {
        self.mesh.faces.is_empty()
    }

    pub fn face_count(&self) -> usize {
        self.mesh.faces.len()
    }

    pub fn iter_points(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.mesh.vertices.iter().map(|v| v.pos)
    }
}

fn nearest_two(points: &[Vec3], pos: Vec3) -> (u32, u32) {
    let count = points.len() as u32;
    let mut indices: Vec<u32> = (0..count - 1).collect();
    indices.sort_by(|&a, &b| {
        let da = (points[a as usize] - pos).length_squared();
        let db = (points[b as usize] - pos).length_squared();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    (indices[0], indices[1])
}

fn point_in_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let n = ab.cross(ac);
    let n_len_sq = n.length_squared();
    if n_len_sq < 1e-12 {
        return false;
    }
    let u = n.cross(ac).dot(ap) / n_len_sq;
    let v = ab.cross(n).dot(ap) / n_len_sq;
    let w = 1.0 - u - v;
    u >= -1e-8 && u <= 1.0 + 1e-8 && v >= -1e-8 && v <= 1.0 + 1e-8 && w >= -1e-8 && w <= 1.0 + 1e-8
}

pub(crate) fn project_to_ndc(p: Vec3, view: Mat4, proj: Mat4) -> Vec2 {
    let clip = proj * view * p.extend(1.0);
    Vec2::new(clip.x / clip.w, clip.y / clip.w)
}

pub(crate) fn ndc_segment_distance(px: f32, py: f32, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-12 {
        return Vec2::new(px - a.x, py - a.y).length();
    }
    let t = ((px - a.x) * ab.x + (py - a.y) * ab.y) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let closest = a + ab * t;
    Vec2::new(px - closest.x, py - closest.y).length()
}
