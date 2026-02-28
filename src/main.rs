use std::{cmp, f32::consts::PI};

use macroquad::{math, prelude::*};

const GRAVITY: f32 = -0.0;
const POINT_RADIUS: f32 = 5.0;
const RESTITUTION: f32 = 0.9;
const BORDER: f32 = 5.0;
const SMOOTHING_RADIUS: f32 = 5.0;
const PIXELS_PER_UNIT: f32 = 100.0;
const TARGET_DENSITY: f32 = 10.0;
const PRESSURE_MULTIPLIER: f32 = 1.0;
const MASS: f32 = 1.0;
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
    velocities: Vec<Vec2>,
    densities: Vec<f32>,
}

impl Simulation {
    fn new_random(count: usize, bounds: &Boundary) -> Self {
        let positions = (0..count)
            .map(|_| {
                Vec2::new(
                    rand::gen_range(bounds.x_min, bounds.x_max),
                    rand::gen_range(bounds.y_min, bounds.y_max),
                )
            })
            .collect();
        let velocities = vec![Vec2::ZERO; count];
        let densities = vec![0.0; count];
        Self {
            positions,
            velocities,
            densities,
        }
    }
    fn new_grid(rows: usize, cols: usize, spacing: f32) -> Self {
        let mut positions = Vec::new();
        let start_x = screen_width() / 2.0 - (cols as f32 / 2.0) * spacing;
        let start_y = screen_height() / 2.0 - (rows as f32 / 2.0) * spacing;
        for row in 0..rows {
            for col in 0..cols {
                positions.push(Vec2::new(
                    start_x + col as f32 * spacing,
                    start_y + row as f32 * spacing,
                ));
            }
        }
        let velocities = vec![Vec2::ZERO; positions.len()];
        let densities = vec![0.0; positions.len()];
        Self {
            positions,
            velocities,
            densities,
        }
    }
    fn update_densities(&mut self, smoothing_radius: f32) {
        for i in 0..self.positions.len() {
            self.densities[i] = self.calculate_density(self.positions[i], smoothing_radius);
        }
    }

    fn update(&mut self, dt: f32, bounds: &Boundary, smoothing_radius: f32) {
        self.update_densities(smoothing_radius);

        for i in 0..self.positions.len() {
            let pressureForce =
                self.calculate_pressure_gradient(self.positions[i], smoothing_radius);
            let pressure_acceleration = pressureForce / self.densities[i];
            self.velocities[i] += pressure_acceleration
        }
        for i in 0..self.positions.len() {
            self.velocities[i].y -= GRAVITY * dt;
            self.positions[i] += self.velocities[i] * dt;
            Simulation::resolve_collisions(&mut self.positions[i], &mut self.velocities[i], bounds);
        }
    }
    fn draw(&self, sample_point: Vec2, smoothing_radius: f32) {
        for pos in &self.positions {
            let dst = (*pos - sample_point).length();
            let color = if dst < smoothing_radius { YELLOW } else { BLUE };
            draw_circle(pos.x, pos.y, POINT_RADIUS, color);
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
    fn resolve_collisions(position: &mut Vec2, velocity: &mut Vec2, bounds: &Boundary) {
        if position.x < bounds.x_min {
            position.x = bounds.x_min;
            velocity.x *= -RESTITUTION;
        } else if position.x > bounds.x_max {
            position.x = bounds.x_max;
            velocity.x *= -RESTITUTION;
        }
        if position.y < bounds.y_min {
            position.y = bounds.y_min;
            velocity.y *= -RESTITUTION;
        } else if position.y > bounds.y_max {
            position.y = bounds.y_max;
            velocity.y *= -RESTITUTION;
        }
    }
    fn smoothing_kernel(radius: f32, dst: f32) -> f32 {
        let volume = PI * radius.powf(8.0) / 4.0;
        let value = f32::max(0.0, radius * radius - dst * dst);
        return value * value * value / volume;
    }
    fn smoothing_kernel_derivative(radius: f32, dst: f32) -> f32 {
        if (dst >= radius) {
            return 0.0;
        }
        let f = radius * radius - dst * dst;
        let scale = -24.0 / (PI * radius.powf(8.0));
        return scale * dst * f * f;
    }
    fn convert_density_to_pressure(density: f32) -> f32 {
        let density_error = density - TARGET_DENSITY;
        let mut pressure = density_error * PRESSURE_MULTIPLIER;
        return pressure;
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
    fn calculate_pressure_gradient(&self, sample_point: Vec2, smoothing_radius: f32) -> Vec2 {
        let mut pressure = Vec2::ZERO;
        let scale = 1.0 / PIXELS_PER_UNIT;
        let scaled_radius = smoothing_radius * scale;

        for i in 0..self.positions.len() {
            let offset = self.positions[i] - sample_point;
            let dst = offset.length() * scale;
            if dst == 0.0 {
                continue;
            }
            let dir = offset.normalize();
            let slope = Simulation::smoothing_kernel_derivative(scaled_radius, dst);
            let density = self.densities[i];
            pressure +=
                -Simulation::convert_density_to_pressure(density) * dir * slope * MASS / density;
        }
        return pressure;
    }
}

#[macroquad::main("FluidSim")]
async fn main() {
    let bounds = Boundary::from_screen();
    let mut sim = Simulation::new_grid(20, 20, 20.0);
    let mut smoothing_radius: f32 = 50.0;
    loop {
        let bounds = Boundary::from_screen();

        let scroll = mouse_wheel().1;
        if scroll != 0.0 {
            smoothing_radius = (smoothing_radius + scroll * 0.05).clamp(5.0, 300.0)
        }
        let mouse_pos: Vec2 = mouse_position().into();

        clear_background(BLACK);

        sim.draw(mouse_pos, smoothing_radius);
        sim.update(get_frame_time(), &bounds, smoothing_radius);

        draw_rectangle_lines(
            5.0,
            5.0,
            screen_width() - 10.0,
            screen_height() - 10.0,
            5.0,
            WHITE,
        );
        next_frame().await
    }
}
