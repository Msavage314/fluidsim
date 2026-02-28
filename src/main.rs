use std::{cmp, f32::consts::PI};

use egui_macroquad::egui;
use macroquad::{math, prelude::*};
use rayon::prelude::*;
const POINT_RADIUS: f32 = 5.0;
const BORDER: f32 = 5.0;
const PIXELS_PER_UNIT: f32 = 100.0;
const MASS: f32 = 1.0;

struct SpatialGrid {
    cell_size: f32,
    particle_indices: Vec<usize>,
    cell_keys: Vec<u32>,
    cell_starts: Vec<u32>,
    table_size: usize,
}
impl SpatialGrid {
    fn new(table_size: usize, cell_size: f32) -> Self {
        Self {
            cell_size,
            particle_indices: Vec::new(),
            cell_keys: Vec::new(),
            cell_starts: vec![u32::MAX; table_size],
            table_size,
        }
    }
    fn cell_coord(&self, pos: Vec2) -> (i32, i32) {
        return (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
        );
    }
    fn hash_cell(&self, cx: i32, cy: i32) -> u32 {
        let a = cx.wrapping_mul(92837111) as u32;
        let b = cy.wrapping_mul(689287499) as u32;
        (a ^ b) % self.table_size as u32
    }

    fn build(&mut self, positions: &[Vec2]) {
        let n = positions.len();
        let mut pairs: Vec<(usize, u32)> = positions
            .iter()
            .enumerate()
            .map(|(i, pos)| {
                let (cx, cy) = self.cell_coord(*pos);
                (i, self.hash_cell(cx, cy))
            })
            .collect();

        pairs.sort_unstable_by_key(|&(_, key)| key);

        self.particle_indices = pairs.iter().map(|&(i, _)| i).collect();
        self.cell_keys = pairs.iter().map(|&(_, k)| k).collect();
        self.cell_starts.fill(u32::MAX);
        for (sorted_pos, &key) in self.cell_keys.iter().enumerate() {
            if sorted_pos == 0 || self.cell_keys[sorted_pos - 1] != key {
                self.cell_starts[key as usize] = sorted_pos as u32;
            }
        }
    }
    fn query_neighbours<'a>(&'a self, pos: Vec2) -> impl Iterator<Item = usize> + 'a {
        let (cx, cy) = self.cell_coord(pos);
        let cell_size = self.cell_size;

        (-1i32..=1)
            .flat_map(move |dx| (-1i32..=1).map(move |dy| (cx + dx, cy + dy)))
            .flat_map(move |(nx, ny)| {
                let key = self.hash_cell(nx, ny) as usize;
                let start = self.cell_starts[key];
                if start == u32::MAX {
                    return vec![];
                }
                let mut result = vec![];
                let mut idx = start as usize;
                while idx < self.cell_keys.len() && self.cell_keys[idx] == key as u32 {
                    result.push(self.particle_indices[idx]);
                    idx += 1;
                }
                result
            })
    }
}

struct Settings {
    gravity: f32,
    smoothing_radius: f32,
    target_density: f32,
    pressure_multiplier: f32,
    restitution: f32,
    speed_scale: f32,
    damping: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            gravity: 0.0,
            smoothing_radius: 50.0,
            target_density: 5.0,
            pressure_multiplier: 5.0,
            restitution: 0.9,
            speed_scale: 300.0,
            damping: 1.0,
        }
    }
}
struct Boundary {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
}

impl Boundary {
    fn from_screen() -> Self {
        Self {
            x_min: BORDER + POINT_RADIUS,
            x_max: screen_width() - BORDER - POINT_RADIUS,
            y_min: BORDER + POINT_RADIUS,
            y_max: screen_height() - BORDER - POINT_RADIUS,
        }
    }
}

struct Simulation {
    positions: Vec<Vec2>,
    predicted_positions: Vec<Vec2>,
    velocities: Vec<Vec2>,
    densities: Vec<f32>,
    settings: Settings,
    grid: SpatialGrid,
}

impl Simulation {
    fn new_random(count: usize, bounds: &Boundary, settings: Settings) -> Self {
        let positions = (0..count)
            .map(|_| {
                Vec2::new(
                    rand::gen_range(bounds.x_min, bounds.x_max),
                    rand::gen_range(bounds.y_min, bounds.y_max),
                )
            })
            .collect();
        let predicted_positions = (0..count)
            .map(|_| {
                Vec2::new(
                    rand::gen_range(bounds.x_min, bounds.x_max),
                    rand::gen_range(bounds.y_min, bounds.y_max),
                )
            })
            .collect();
        let velocities = vec![Vec2::ZERO; count];
        let densities = vec![0.0; count];
        let grid = SpatialGrid::new(count * 4, settings.smoothing_radius);
        Self {
            positions,
            predicted_positions,
            velocities,
            densities,
            settings,
            grid,
        }
    }
    fn new_grid(rows: usize, cols: usize, spacing: f32, settings: Settings) -> Self {
        let mut positions = Vec::new();
        let mut predicted_positions = Vec::new();
        let start_x = screen_width() / 2.0 - (cols as f32 / 2.0) * spacing;
        let start_y = screen_height() / 2.0 - (rows as f32 / 2.0) * spacing;
        for row in 0..rows {
            for col in 0..cols {
                positions.push(Vec2::new(
                    start_x + col as f32 * spacing,
                    start_y + row as f32 * spacing,
                ));
                predicted_positions.push(Vec2::new(
                    start_x + col as f32 * spacing,
                    start_y + row as f32 * spacing,
                ));
            }
        }
        let count = positions.len();
        let velocities = vec![Vec2::ZERO; count];
        let densities = vec![0.0; count];
        let grid = SpatialGrid::new(count * 4, settings.smoothing_radius);
        Self {
            positions,
            predicted_positions,
            velocities,
            densities,
            settings,
            grid,
        }
    }
    fn update_densities(&mut self, smoothing_radius: f32) {
        let grid = &self.grid;
        self.densities
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, density)| {
                *density = Simulation::calculate_density_static(
                    &self.predicted_positions,
                    grid,
                    smoothing_radius,
                    self.predicted_positions[i],
                );
            });
    }
    fn update_pressure(&mut self, dt: f32, smoothing_radius: f32) {
        let densities = &self.densities;
        let settings = &self.settings;

        let accelerations: Vec<Vec2> = (0..self.predicted_positions.len())
            .into_par_iter()
            .map(|i| {
                let force = Simulation::calculate_pressure_gradient_static(
                    &self.predicted_positions,
                    densities,
                    &self.grid,
                    settings,
                    i,
                    smoothing_radius,
                );
                if densities[i] > 0.001 {
                    force / densities[i]
                } else {
                    Vec2::ZERO
                }
            })
            .collect();

        for i in 0..self.velocities.len() {
            self.velocities[i] -= accelerations[i] * dt;
        }
    }
    fn update(&mut self, dt: f32, bounds: &Boundary, smoothing_radius: f32) {
        for i in 0..self.positions.len() {
            self.predicted_positions[i] = self.positions[i] + self.velocities[i] * dt;
        }

        self.grid.cell_size = self.settings.smoothing_radius;
        self.grid.build(&self.predicted_positions);
        self.update_densities(smoothing_radius);
        self.update_pressure(dt, smoothing_radius);

        for i in 0..self.positions.len() {
            self.velocities[i].y -= self.settings.gravity * dt;
            self.velocities[i] *= self.settings.damping;
            self.positions[i] += self.velocities[i] * dt;
            Simulation::resolve_collisions(
                &mut self.positions[i],
                &mut self.velocities[i],
                bounds,
                self.settings.restitution,
            );
        }
    }
    fn calculate_density_static(
        positions: &[Vec2],
        grid: &SpatialGrid,
        smoothing_radius: f32,
        sample_point: Vec2,
    ) -> f32 {
        let scale = 1.0 / PIXELS_PER_UNIT;
        let scaled_radius = smoothing_radius * scale;

        grid.query_neighbours(sample_point).fold(0.0, |density, i| {
            let dst = (positions[i] * scale - sample_point * scale).length();
            density + MASS * Simulation::smoothing_kernel(scaled_radius, dst)
        })
    }
    fn calculate_pressure_gradient_static(
        positions: &[Vec2],
        densities: &[f32],
        grid: &SpatialGrid,
        settings: &Settings,
        sample_point_index: usize,
        smoothing_radius: f32,
    ) -> Vec2 {
        let scale = 1.0 / PIXELS_PER_UNIT;
        let scaled_radius = smoothing_radius * scale;
        let sample_pos = positions[sample_point_index];
        let sample_density = densities[sample_point_index];

        grid.query_neighbours(sample_pos)
            .fold(Vec2::ZERO, |pressure, i| {
                let offset = positions[i] - sample_pos;
                let dst = offset.length() * scale;
                if dst == 0.0 || densities[i] < 0.001 {
                    return pressure;
                }
                let dir = offset.normalize();
                let slope = Simulation::smoothing_kernel_derivative(scaled_radius, dst);
                let pressure_a =
                    (densities[i] - settings.target_density) * settings.pressure_multiplier;
                let pressure_b =
                    (sample_density - settings.target_density) * settings.pressure_multiplier;
                let shared_pressure = (pressure_a + pressure_b) / 2.0;
                pressure + (-shared_pressure * dir * slope * MASS / densities[i])
            })
    }
    fn draw(&self, sample_point: Vec2, smoothing_radius: f32) {
        for i in 0..self.positions.len() {
            let speed = self.velocities[i].length();
            let t = (speed / self.settings.speed_scale).clamp(0.0, 1.0);
            let hue = (1.0 - t) * 240.0 / 360.0; // 240° (blue) down to 0° (red)
            let color = macroquad::color::hsl_to_rgb(hue, 1.0, 0.5);
            draw_circle(
                self.positions[i].x,
                self.positions[i].y,
                POINT_RADIUS,
                color,
            );
        }
        draw_circle_lines(sample_point.x, sample_point.y, smoothing_radius, 1.0, GREEN);

        let density = self.calculate_density(sample_point, smoothing_radius);
        draw_text(
            &format!("Density: {:.4}  Radius: {:.1}", density, smoothing_radius),
            10.0,
            screen_height() - 10.0,
            20.0,
            WHITE,
        );
    }
    fn resolve_collisions(
        position: &mut Vec2,
        velocity: &mut Vec2,
        bounds: &Boundary,
        restitution: f32,
    ) {
        if position.x < bounds.x_min {
            position.x = bounds.x_min;
            velocity.x *= -restitution;
        } else if position.x > bounds.x_max {
            position.x = bounds.x_max;
            velocity.x *= -restitution;
        }
        if position.y < bounds.y_min {
            position.y = bounds.y_min;
            velocity.y *= -restitution;
        } else if position.y > bounds.y_max {
            position.y = bounds.y_max;
            velocity.y *= -restitution;
        }
    }
    fn smoothing_kernel(radius: f32, dst: f32) -> f32 {
        if dst >= radius {
            return 0.0;
        }
        let volume = PI * radius.powf(4.0) / 6.0;
        return (radius - dst) * (radius - dst) / volume;
    }
    fn smoothing_kernel_derivative(radius: f32, dst: f32) -> f32 {
        if (dst >= radius) {
            return 0.0;
        }
        let scale = 12.0 / (PI * radius.powf(4.0));
        return (dst - radius) * scale;
    }

    fn calculate_density(&self, sample_point: Vec2, smoothing_radius: f32) -> f32 {
        let scale = 1.0 / PIXELS_PER_UNIT;
        let scaled_sample = sample_point * scale;
        let scaled_radius = smoothing_radius * scale;

        self.positions.iter().fold(0.0, |density, pos| {
            let dst = (*pos * scale - scaled_sample).length();
            density + MASS * Simulation::smoothing_kernel(scaled_radius, dst)
        })
    }
}

#[macroquad::main("FluidSim")]
async fn main() {
    let bounds = Boundary::from_screen();
    let mut settings = Settings::default();
    let mut sim = Simulation::new_random(5000, &bounds, settings);
    loop {
        let bounds = Boundary::from_screen();

        let scroll = mouse_wheel().1;
        if scroll != 0.0 {
            sim.settings.smoothing_radius =
                (sim.settings.smoothing_radius + scroll * 0.05).clamp(5.0, 300.0)
        }
        let mouse_pos: Vec2 = mouse_position().into();

        clear_background(BLACK);

        sim.draw(mouse_pos, sim.settings.smoothing_radius);
        sim.update(get_frame_time(), &bounds, sim.settings.smoothing_radius);

        draw_rectangle_lines(
            5.0,
            5.0,
            screen_width() - 10.0,
            screen_height() - 10.0,
            5.0,
            WHITE,
        );

        egui_macroquad::ui(|ctx| {
            egui::Window::new("Settings").show(ctx, |ui| {
                ui.add(
                    egui::Slider::new(&mut sim.settings.gravity, -500.0..=500.0).text("Gravity"),
                );
                ui.add(
                    egui::Slider::new(&mut sim.settings.smoothing_radius, 5.0..=300.0)
                        .text("Smoothing Radius"),
                );
                ui.add(
                    egui::Slider::new(&mut sim.settings.target_density, 0.0..=50.0)
                        .text("Target Density"),
                );
                ui.add(
                    egui::Slider::new(&mut sim.settings.pressure_multiplier, 0.0..=500.0)
                        .text("Pressure Multiplier"),
                );
                ui.add(
                    egui::Slider::new(&mut sim.settings.restitution, 0.0..=1.0).text("Restitution"),
                );
                ui.add(
                    egui::Slider::new(&mut sim.settings.speed_scale, 10.0..=1000.0)
                        .text("Speed Colour Scale"),
                );
                ui.add(egui::Slider::new(&mut sim.settings.damping, 0.0..=1.0).text("Damping"));
            });
        });
        egui_macroquad::draw();
        next_frame().await
    }
}
