#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::{
    egui::{self, Color32, Key, Layout, PointerButton, Pos2, Rect, RichText, Sense, Stroke, TextureOptions, Vec2, Window},
    epaint::{ColorImage, TextureHandle},
};
use image::{DynamicImage, Rgba, RgbaImage};
use rand::seq::SliceRandom;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

// --- Tiling Configuration ---
const TILE_WIDTH: u32 = 256;
const TILE_HEIGHT: u32 = 256;
const TILE_STRIDE: u32 = 204;

// --- Configuration Structs ---

#[derive(Serialize, Deserialize, Clone, Debug)]
struct BrushDef {
    name: String,
    color: [u8; 4], // RGBA
    shortcut_key: Option<String>,
}

impl BrushDef {
    fn color32(&self) -> Color32 {
        let [r, g, b, a] = self.color;
        if a == 0 {
            Color32::WHITE
        } else {
            Color32::from_rgba_unmultiplied(r, g, b, a)
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct AppConfig {
    brushes: Vec<BrushDef>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            brushes: vec![
                BrushDef { name: "Eraser".to_string(), color: [0, 0, 0, 0], shortcut_key: Some("E".to_string()) },
                BrushDef { name: "Hole".to_string(), color: [227, 26, 28, 255], shortcut_key: Some("H".to_string()) },
                BrushDef { name: "Mining".to_string(), color: [254, 13, 240, 255], shortcut_key: Some("M".to_string()) },
                BrushDef { name: "Skeletonizer".to_string(), color: [106, 61, 154, 255], shortcut_key: Some("S".to_string()) },
                BrushDef { name: "Surface".to_string(), color: [218, 137, 55, 255], shortcut_key: Some("O".to_string()) },
                BrushDef { name: "Anomaly".to_string(), color: [128, 128, 128, 255], shortcut_key: Some("A".to_string()) },
            ],
        }
    }
}

// --- State Structs ---

#[derive(PartialEq)]
enum AppTab {
    Classifier,
    Segmentation,
}

struct ClassificationAction {
    original_source: PathBuf,
    copied_destination: PathBuf,
    label: String,
}

struct ContextMap {
    texture: Option<TextureHandle>,
    base_name: String,
    active_rect: Option<Rect>,
    pan: Vec2,
    scale: f32,
}

struct ClassifierState {
    workspace_path: Option<PathBuf>,
    current_image_path: Option<PathBuf>,
    current_texture: Option<TextureHandle>,
    
    tile_pan: Vec2,
    tile_scale: f32,

    context_map: ContextMap,
    filename_regex: Regex,
    last_classified_texture: Option<TextureHandle>,
    last_classified_label: String,
    processed_session_files: HashSet<PathBuf>,
    undo_stack: Vec<ClassificationAction>,
    session_count: usize,
    status_msg: String,
}

struct SegmentationState {
    images_queue: Vec<PathBuf>,
    current_image_path: Option<PathBuf>,
    base_texture: Option<TextureHandle>,
    mask_texture: Option<TextureHandle>,
    mask_image: Option<RgbaImage>,
    
    active_brush_index: usize,
    brush_size: f32,
    pan: Vec2,
    scale: f32,
}

struct MyApp {
    current_tab: AppTab,
    config: AppConfig,
    show_settings: bool,
    classifier: ClassifierState,
    segmenter: SegmentationState,
}

impl MyApp {
    fn load_config() -> AppConfig {
        if let Ok(content) = fs::read_to_string("config.json") {
            if let Ok(cfg) = serde_json::from_str(&content) {
                return cfg;
            }
        }
        AppConfig::default()
    }

    // FIX 1: Make this static and take &AppConfig to avoid borrowing 'self' completely
    fn save_config(config: &AppConfig) {
        if let Ok(json) = serde_json::to_string_pretty(config) {
            let _ = fs::write("config.json", json);
        }
    }
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            current_tab: AppTab::Classifier,
            config: MyApp::load_config(),
            show_settings: false,
            classifier: ClassifierState {
                workspace_path: None,
                current_image_path: None,
                current_texture: None,
                tile_pan: Vec2::ZERO,
                tile_scale: 1.0,
                context_map: ContextMap {
                    texture: None,
                    base_name: String::new(),
                    active_rect: None,
                    pan: Vec2::ZERO,
                    scale: 0.5,
                },
                filename_regex: Regex::new(r"^(.*)_(\d+)_(\d+)\.[^.]+$").unwrap(),
                last_classified_texture: None,
                last_classified_label: String::new(),
                processed_session_files: HashSet::new(),
                undo_stack: Vec::new(),
                session_count: 0,
                status_msg: "Please load a Workspace Folder.".to_owned(),
            },
            segmenter: SegmentationState {
                images_queue: Vec::new(),
                current_image_path: None,
                base_texture: None,
                mask_texture: None,
                mask_image: None,
                active_brush_index: 1,
                brush_size: 20.0,
                pan: Vec2::ZERO,
                scale: 1.0,
            },
        }
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 900.0])
            .with_title("Image Classifier & Segmenter"),
        ..Default::default()
    };
    eframe::run_native(
        "Image Workstation",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    )
}

fn str_to_key(s: &str) -> Option<Key> {
    match s.to_uppercase().as_str() {
        "A" => Some(Key::A), "B" => Some(Key::B), "C" => Some(Key::C),
        "D" => Some(Key::D), "E" => Some(Key::E), "F" => Some(Key::F),
        "G" => Some(Key::G), "H" => Some(Key::H), "I" => Some(Key::I),
        "J" => Some(Key::J), "K" => Some(Key::K), "L" => Some(Key::L),
        "M" => Some(Key::M), "N" => Some(Key::N), "O" => Some(Key::O),
        "P" => Some(Key::P), "Q" => Some(Key::Q), "R" => Some(Key::R),
        "S" => Some(Key::S), "T" => Some(Key::T), "U" => Some(Key::U),
        "V" => Some(Key::V), "W" => Some(Key::W), "X" => Some(Key::X),
        "Y" => Some(Key::Y), "Z" => Some(Key::Z),
        "0" => Some(Key::Num0), "1" => Some(Key::Num1), "2" => Some(Key::Num2),
        "3" => Some(Key::Num3), "4" => Some(Key::Num4), "5" => Some(Key::Num5),
        "6" => Some(Key::Num6), "7" => Some(Key::Num7), "8" => Some(Key::Num8),
        "9" => Some(Key::Num9),
        _ => None,
    }
}

impl MyApp {
    fn parse_tile_info(&self, path: &Path) -> Option<(String, u32, u32)> {
        let filename = path.file_name()?.to_string_lossy();
        if let Some(caps) = self.classifier.filename_regex.captures(&filename) {
            let base = caps.get(1)?.as_str().to_string();
            let idx_x = caps.get(2)?.as_str().parse::<u32>().ok()?;
            let idx_y = caps.get(3)?.as_str().parse::<u32>().ok()?;
            return Some((base, idx_x, idx_y));
        }
        None
    }

    fn load_image_to_texture(
        ctx: &egui::Context,
        path: &Path,
    ) -> Result<(TextureHandle, DynamicImage), String> {
        let img = image::open(path).map_err(|e| e.to_string())?;
        let size = [img.width() as _, img.height() as _];
        let image_buffer = img.to_rgba8();
        let pixels = image_buffer.as_flat_samples();
        let color_image = ColorImage::from_rgba_unmultiplied(size, pixels.as_slice());
        Ok((
            ctx.load_texture(
                path.to_string_lossy().to_string(),
                color_image,
                TextureOptions::LINEAR,
            ),
            img,
        ))
    }

    fn rebuild_context_map(&mut self, ctx: &egui::Context, current_tile_path: &Path) {
        let ws = match &self.classifier.workspace_path {
            Some(p) => p,
            None => return,
        };

        let (base_name, cur_idx_x, cur_idx_y) = match self.parse_tile_info(current_tile_path) {
            Some(info) => info,
            None => {
                self.classifier.context_map.texture = None;
                return;
            }
        };

        let pixel_x = (cur_idx_x.saturating_sub(1) * TILE_STRIDE) as f32;
        let pixel_y = (cur_idx_y.saturating_sub(1) * TILE_STRIDE) as f32;

        self.classifier.context_map.active_rect = Some(Rect::from_min_size(
            Pos2::new(pixel_x, pixel_y),
            Vec2::new(TILE_WIDTH as f32, TILE_HEIGHT as f32)
        ));

        if self.classifier.context_map.base_name == base_name && self.classifier.context_map.texture.is_some() {
            return;
        }

        let mut siblings: Vec<(PathBuf, u32, u32)> = Vec::new();
        if let Ok(entries) = fs::read_dir(ws) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() { continue; }
                if let Some(ext) = path.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "tiff") {
                        if let Some((b, x, y)) = self.parse_tile_info(&path) {
                            if b == base_name {
                                siblings.push((path, x, y));
                            }
                        }
                    }
                }
            }
        }

        if siblings.is_empty() { return; }

        let max_idx_x = siblings.iter().map(|(_, x, _)| *x).max().unwrap_or(1);
        let max_idx_y = siblings.iter().map(|(_, _, y)| *y).max().unwrap_or(1);

        let canvas_w = (max_idx_x - 1) * TILE_STRIDE + TILE_WIDTH;
        let canvas_h = (max_idx_y - 1) * TILE_STRIDE + TILE_HEIGHT;

        if canvas_w > 16384 || canvas_h > 16384 {
            self.classifier.status_msg = "Context map too large to generate.".to_string();
            self.classifier.context_map.texture = None;
            return;
        }

        let mut canvas = RgbaImage::new(canvas_w, canvas_h);
        siblings.sort_by_key(|k| (k.2, k.1));

        for (path, idx_x, idx_y) in siblings {
            if let Ok(img) = image::open(&path) {
                let rgba = img.to_rgba8();
                let target_x = (idx_x - 1) * TILE_STRIDE;
                let target_y = (idx_y - 1) * TILE_STRIDE;
                image::imageops::overlay(&mut canvas, &rgba, target_x as i64, target_y as i64);
            }
        }

        let size = [canvas_w as _, canvas_h as _];
        let pixels = canvas.as_flat_samples();
        let color_image = ColorImage::from_rgba_unmultiplied(size, pixels.as_slice());
        let tex = ctx.load_texture(format!("ctx_{}", base_name), color_image, TextureOptions::LINEAR);

        self.classifier.context_map.texture = Some(tex);
        self.classifier.context_map.base_name = base_name;
        self.classifier.context_map.pan = Vec2::ZERO;
        self.classifier.context_map.scale = 0.5;
    }

    fn load_random_classifier_image(&mut self, ctx: &egui::Context) {
        if let Some(ws) = &self.classifier.workspace_path {
            let all_files: Vec<PathBuf> = fs::read_dir(ws)
                .ok()
                .map(|iter| {
                    iter.filter_map(|entry| entry.ok())
                        .map(|e| e.path())
                        .filter(|p| p.is_file())
                        .collect()
                })
                .unwrap_or_default();

            let candidates: Vec<&PathBuf> = all_files
                .iter()
                .filter(|p| {
                     if let Some(ext) = p.extension() {
                         let s = ext.to_string_lossy().to_lowercase();
                         matches!(s.as_str(), "jpg"|"jpeg"|"png"|"bmp"|"tiff")
                     } else { false }
                })
                .filter(|p| !self.classifier.processed_session_files.contains(*p))
                .collect();

            if let Some(random_file) = candidates.choose(&mut rand::thread_rng()) {
                match Self::load_image_to_texture(ctx, random_file) {
                    Ok((tex, _)) => {
                        self.classifier.current_texture = Some(tex);
                        self.classifier.current_image_path = Some((*random_file).clone());
                        self.classifier.status_msg =
                            format!("Loaded: {:?}", random_file.file_name().unwrap());
                        self.classifier.tile_pan = Vec2::ZERO;
                        self.classifier.tile_scale = 1.0;
                        
                        self.rebuild_context_map(ctx, random_file);
                    }
                    Err(e) => self.classifier.status_msg = format!("Error loading image: {}", e),
                }
            } else {
                self.classifier.current_texture = None;
                self.classifier.current_image_path = None;
                self.classifier.context_map.texture = None;
                self.classifier.status_msg = if all_files.is_empty() {
                    "No images found in folder.".to_string()
                } else {
                    "All images classified for this session!".to_string()
                };
            }
        }
    }

    fn classify_current(&mut self, ctx: &egui::Context, label: &str) {
        if let (Some(src), Some(ws)) = (
            &self.classifier.current_image_path.clone(),
            &self.classifier.workspace_path,
        ) {
            let dest_folder = ws.join(label);
            if !dest_folder.exists() {
                let _ = fs::create_dir_all(&dest_folder);
            }

            let file_name = src.file_name().unwrap();
            let dest_path = dest_folder.join(file_name);

            match fs::copy(src, &dest_path) {
                Ok(_) => {
                    self.classifier.processed_session_files.insert(src.clone());
                    self.classifier.session_count += 1;

                    self.classifier.undo_stack.push(ClassificationAction {
                        original_source: src.clone(),
                        copied_destination: dest_path.clone(),
                        label: label.to_string(),
                    });

                    self.classifier.last_classified_texture =
                        self.classifier.current_texture.take();
                    self.classifier.last_classified_label = label.to_string();

                    self.load_random_classifier_image(ctx);
                }
                Err(e) => {
                    self.classifier.status_msg = format!("Copy failed: {}", e);
                }
            }
        }
    }

    fn undo_last_classification(&mut self, ctx: &egui::Context) {
        if let Some(action) = self.classifier.undo_stack.pop() {
            if action.copied_destination.exists() {
                let _ = fs::remove_file(&action.copied_destination);
            }

            self.classifier.processed_session_files.remove(&action.original_source);
            
            if self.classifier.session_count > 0 {
                self.classifier.session_count -= 1;
            }

            match Self::load_image_to_texture(ctx, &action.original_source) {
                Ok((tex, _)) => {
                    self.classifier.current_texture = Some(tex);
                    self.classifier.current_image_path = Some(action.original_source.clone());
                    self.classifier.status_msg = format!("Undid: {}", action.label);
                    self.rebuild_context_map(ctx, &action.original_source);
                }
                Err(_) => self.classifier.status_msg = "Error reloading undid image".to_string(),
            }

            if let Some(prev_action) = self.classifier.undo_stack.last() {
                 match Self::load_image_to_texture(ctx, &prev_action.original_source) {
                    Ok((tex, _)) => {
                        self.classifier.last_classified_texture = Some(tex);
                        self.classifier.last_classified_label = prev_action.label.clone();
                    }
                    Err(_) => {
                        self.classifier.last_classified_texture = None;
                    }
                 }
            } else {
                self.classifier.last_classified_texture = None;
                self.classifier.last_classified_label.clear();
            }
        }
    }

    fn init_segmentation_queue(&mut self) {
        if let Some(ws) = &self.classifier.workspace_path {
            let damaged_dir = ws.join("Damaged");
            let mask_dir = ws.join("masks");

            if !damaged_dir.exists() {
                self.segmenter.images_queue.clear();
                return;
            }

            let all_damaged: Vec<PathBuf> = fs::read_dir(damaged_dir)
                .ok()
                .map(|iter| {
                    iter.filter_map(|entry| entry.ok())
                        .map(|e| e.path())
                        .filter(|p| {
                            if let Some(ext) = p.extension() {
                                let s = ext.to_string_lossy().to_lowercase();
                                matches!(s.as_str(), "jpg"|"jpeg"|"png"|"bmp"|"tiff")
                            } else { false }
                        })
                        .collect()
                })
                .unwrap_or_default();

            self.segmenter.images_queue = all_damaged.into_iter().filter(|p| {
                let fname = p.file_name().unwrap();
                let mask_path = mask_dir.join(fname);
                !mask_path.exists()
            }).collect();

            self.segmenter.images_queue.shuffle(&mut rand::thread_rng());
        }
    }

    fn load_next_segmentation(&mut self, ctx: &egui::Context) {
        if let Some(path) = self.segmenter.images_queue.pop() {
            match Self::load_image_to_texture(ctx, &path) {
                Ok((tex, img)) => {
                    self.segmenter.current_image_path = Some(path);
                    self.segmenter.base_texture = Some(tex);
                    
                    let mask = RgbaImage::new(img.width(), img.height());
                    self.segmenter.mask_image = Some(mask);
                    self.update_mask_texture(ctx);

                    self.segmenter.pan = Vec2::ZERO;
                    self.segmenter.scale = 1.0;
                }
                Err(_) => {}
            }
        } else {
            self.segmenter.base_texture = None;
            self.segmenter.mask_texture = None;
            self.segmenter.current_image_path = None;
        }
    }

    fn update_mask_texture(&mut self, ctx: &egui::Context) {
        if let Some(mask_img) = &self.segmenter.mask_image {
            let size = [mask_img.width() as _, mask_img.height() as _];
            let pixels = mask_img.as_flat_samples();
            let color_image = ColorImage::from_rgba_unmultiplied(size, pixels.as_slice());
            self.segmenter.mask_texture =
                Some(ctx.load_texture("mask", color_image, TextureOptions::NEAREST));
        }
    }

    fn save_mask(&self) {
        if let (Some(path), Some(mask), Some(ws)) = (
            &self.segmenter.current_image_path,
            &self.segmenter.mask_image,
            &self.classifier.workspace_path,
        ) {
            let mask_folder = ws.join("masks");
            if !mask_folder.exists() {
                let _ = fs::create_dir_all(&mask_folder);
            }
            let filename = path.file_name().unwrap();
            let save_path = mask_folder.join(filename);

            let mut final_save_img = mask.clone();
            for pixel in final_save_img.pixels_mut() {
                if pixel[3] == 0 {
                    *pixel = Rgba([0, 0, 0, 255]);
                }
            }
            let _ = final_save_img.save(save_path);
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- GLOBAL INPUTS ---
        if self.current_tab == AppTab::Classifier {
             if ctx.input(|i| i.key_pressed(Key::Num1)) { self.classify_current(ctx, "Healthy"); }
             if ctx.input(|i| i.key_pressed(Key::Num2)) { self.classify_current(ctx, "Undecided"); }
             if ctx.input(|i| i.key_pressed(Key::Num3)) { self.classify_current(ctx, "Damaged"); }
        } else if self.current_tab == AppTab::Segmentation {
            if ctx.input(|i| i.key_pressed(Key::Plus) || i.key_pressed(Key::Equals)) { self.segmenter.brush_size += 2.0; }
            if ctx.input(|i| i.key_pressed(Key::Minus)) { self.segmenter.brush_size = (self.segmenter.brush_size - 2.0).max(1.0); }
            
            for (i, brush) in self.config.brushes.iter().enumerate() {
                if let Some(key_str) = &brush.shortcut_key {
                    if let Some(key) = str_to_key(key_str) {
                         if ctx.input(|inp| inp.key_pressed(key)) {
                             self.segmenter.active_brush_index = i;
                        }
                    }
                }
            }
        }

        // --- SETTINGS WINDOW ---
        if self.show_settings {
            let mut open = true;
            let mut should_close = false; // Flag to handle closing from inside

            Window::new("Settings").open(&mut open).show(ctx, |ui| {
                ui.heading("Brush Configuration");
                ui.separator();
                
                let mut to_remove = None;
                for (i, brush) in self.config.brushes.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.text_edit_singleline(&mut brush.name);
                        ui.color_edit_button_srgba_unmultiplied(&mut brush.color);
                        if ui.button("🗑").clicked() {
                            to_remove = Some(i);
                        }
                    });
                }
                
                if let Some(idx) = to_remove {
                    if self.config.brushes.len() > 1 {
                        self.config.brushes.remove(idx);
                        if self.segmenter.active_brush_index >= self.config.brushes.len() {
                            self.segmenter.active_brush_index = 0;
                        }
                    }
                }

                if ui.button("+ Add Brush").clicked() {
                    self.config.brushes.push(BrushDef {
                        name: "New".to_string(),
                        color: [255, 255, 255, 255],
                        shortcut_key: None,
                    });
                }
                
                ui.separator();
                if ui.button("Save & Close").clicked() {
                    Self::save_config(&self.config);
                    should_close = true; // Signal to close
                }
            });

            // Update state after the closure is done (borrow ends)
            if should_close {
                open = false;
            }
            self.show_settings = open;
        }
        
        // --- TOP PANEL ---
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.current_tab, AppTab::Classifier, " 1. Classifier ");
                if ui.selectable_value(&mut self.current_tab, AppTab::Segmentation, " 2. Segmentation ").clicked() {
                    if self.segmenter.images_queue.is_empty() && self.segmenter.base_texture.is_none() {
                        self.init_segmentation_queue();
                        self.load_next_segmentation(ctx);
                    }
                }
                
                if ui.button("⚙ Settings").clicked() {
                    self.show_settings = !self.show_settings;
                }

                ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.current_tab == AppTab::Classifier {
                        ui.label(RichText::new(format!("Session: {}", self.classifier.session_count)).strong());
                    } else {
                        ui.label(format!("Queue: {}", self.segmenter.images_queue.len()));
                    }
                });
            });
        });

        match self.current_tab {
            AppTab::Classifier => self.ui_classifier(ctx),
            AppTab::Segmentation => self.ui_segmentation(ctx),
        }
    }
}

impl MyApp {
    // --- ZOOM/PAN HELPER ---
    fn ui_zoomable_image(ui: &mut egui::Ui, tex: &TextureHandle, pan: &mut Vec2, scale: &mut f32, max_height: Option<f32>, overlay_rect: Option<Rect>) {
        let avail_size = ui.available_size();
        let height = max_height.unwrap_or(avail_size.y);
        
        let (rect, response) = ui.allocate_exact_size(Vec2::new(avail_size.x, height), Sense::click_and_drag());
        
        if response.hovered() {
             let scroll = ui.input(|i| i.raw_scroll_delta);
             if scroll.y != 0.0 {
                 let factor = if scroll.y > 0.0 { 1.1 } else { 0.9 };
                 *scale *= factor;
             }
        }
        if response.dragged_by(PointerButton::Middle) || (response.dragged() && ui.input(|i| i.modifiers.shift)) {
            *pan += response.drag_delta();
        }

        let img_w = tex.size()[0] as f32 * *scale;
        let img_h = tex.size()[1] as f32 * *scale;
        let center = rect.center() + *pan;
        let image_rect = Rect::from_center_size(center, [img_w, img_h].into());

        let painter = ui.painter().with_clip_rect(rect);
        painter.image(tex.id(), image_rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);

        if let Some(active_rect_px) = overlay_rect {
             let tex_w = tex.size()[0] as f32;
             let tex_h = tex.size()[1] as f32;
             
             let ratio_x = image_rect.width() / tex_w;
             let ratio_y = image_rect.height() / tex_h;
             
             let screen_x = image_rect.min.x + active_rect_px.min.x * ratio_x;
             let screen_y = image_rect.min.y + active_rect_px.min.y * ratio_y;
             let screen_w = active_rect_px.width() * ratio_x;
             let screen_h = active_rect_px.height() * ratio_y;
             
             painter.rect_stroke(
                Rect::from_min_size(Pos2::new(screen_x, screen_y), Vec2::new(screen_w, screen_h)), 
                0.0, 
                Stroke::new(3.0, Color32::RED)
            );
        }
    }

    fn ui_classifier(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("class_bottom_panel")
            .resizable(false)
            .exact_height(100.0)
            .show(ctx, |ui| {
                ui.with_layout(egui::Layout::centered_and_justified(egui::Direction::LeftToRight), |ui| {
                     ui.horizontal(|ui| {
                        let btn_size = Vec2::new(140.0, 50.0);
                        let has_image = self.classifier.current_texture.is_some();
                        
                        let big_btn = |ui: &mut egui::Ui, text: &str, color: Color32| {
                            ui.add_enabled(has_image, egui::Button::new(RichText::new(text).size(18.0).strong().color(Color32::WHITE)).min_size(btn_size).fill(color))
                        };

                        if big_btn(ui, "Healthy (1)", Color32::from_rgb(46, 125, 50)).clicked() { self.classify_current(ctx, "Healthy"); }
                        ui.add_space(15.0);
                        if big_btn(ui, "Undecided (2)", Color32::from_rgb(117, 117, 117)).clicked() { self.classify_current(ctx, "Undecided"); }
                        ui.add_space(15.0);
                        if big_btn(ui, "Damaged (3)", Color32::from_rgb(198, 40, 40)).clicked() { self.classify_current(ctx, "Damaged"); }

                        ui.add_space(40.0);
                        ui.separator();
                        ui.add_space(40.0);

                        if ui.add_enabled(!self.classifier.undo_stack.is_empty(), egui::Button::new(RichText::new("Undo").size(18.0)).min_size(btn_size)).clicked() {
                            self.undo_last_classification(ctx);
                        }
                    });
                });
            });

        egui::SidePanel::left("class_left_panel")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.add_space(10.0);
                    ui.heading("Setup");
                    if ui.button("Load Workspace Folder").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                            self.classifier.workspace_path = Some(path);
                            self.classifier.undo_stack.clear();
                            self.classifier.processed_session_files.clear();
                            self.classifier.session_count = 0;
                            self.classifier.last_classified_texture = None;
                            self.classifier.context_map.texture = None;
                            self.load_random_classifier_image(ctx);
                        }
                    }
                    ui.label(RichText::new(&self.classifier.status_msg).size(12.0).italics());

                    ui.separator();
                    ui.heading("Last Classified");
                    if let Some(tex) = &self.classifier.last_classified_texture {
                        ui.label(RichText::new(&self.classifier.last_classified_label).strong().size(16.0));
                        ui.add(egui::Image::new(tex).max_width(200.0));
                    } else {
                        ui.label("No history.");
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Context Map
            // FIX: Clone texture handle to unborrow 'self'
            let ctx_tex_handle = self.classifier.context_map.texture.clone();
            
            if let Some(ctx_tex) = ctx_tex_handle {
                ui.vertical(|ui| {
                    ui.label(RichText::new(format!("Context: {}", self.classifier.context_map.base_name)).strong());
                    
                    let avail_h = ui.available_height();
                    let map_h = if self.classifier.current_texture.is_some() { avail_h * 0.45 } else { avail_h };

                    Self::ui_zoomable_image(
                        ui, 
                        &ctx_tex, 
                        &mut self.classifier.context_map.pan, 
                        &mut self.classifier.context_map.scale, 
                        Some(map_h), 
                        self.classifier.context_map.active_rect
                    );
                });
                ui.separator();
            }

            // Current Tile
            let curr_tex_handle = self.classifier.current_texture.clone();
            if let Some(tex) = curr_tex_handle {
                Self::ui_zoomable_image(
                    ui, 
                    &tex, 
                    &mut self.classifier.tile_pan, 
                    &mut self.classifier.tile_scale, 
                    None, 
                    None
                );
            } else {
                ui.centered_and_justified(|ui| { ui.heading("No Image Loaded"); });
            }
        });
    }

    fn ui_segmentation(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("seg_tools_panel").show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Tools");
                ui.separator();
                
                for (i, brush) in self.config.brushes.iter().enumerate() {
                    let is_selected = self.segmenter.active_brush_index == i;
                    let txt = match &brush.shortcut_key {
                        Some(k) => format!("{} ({})", brush.name, k),
                        None => brush.name.clone(),
                    };
                    if ui.selectable_label(is_selected, RichText::new(txt).color(brush.color32()).strong()).clicked() {
                        self.segmenter.active_brush_index = i;
                    }
                }

                ui.separator();
                ui.label(format!("Size: {:.0}", self.segmenter.brush_size));
                ui.add(egui::Slider::new(&mut self.segmenter.brush_size, 1.0..=100.0));

                ui.add_space(20.0);
                if ui.button("Reset Mask").clicked() {
                    if let Some(mask) = &mut self.segmenter.mask_image {
                        *mask = RgbaImage::new(mask.width(), mask.height());
                        self.update_mask_texture(ctx);
                    }
                }

                ui.add_space(10.0);
                if ui.button("Save & Next").clicked() {
                    self.save_mask();
                    self.load_next_segmentation(ctx);
                }
                if ui.button("Skip").clicked() {
                    self.load_next_segmentation(ctx);
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let (Some(base_tex), Some(mask_tex)) = (&self.segmenter.base_texture, &self.segmenter.mask_texture) {
                let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());

                // Zoom/Pan
                let scroll = ctx.input(|i| i.raw_scroll_delta);
                if scroll.y != 0.0 {
                    let factor = if scroll.y > 0.0 { 1.1 } else { 0.9 };
                    self.segmenter.scale *= factor;
                }
                if response.dragged_by(PointerButton::Middle) || (response.dragged() && ctx.input(|i| i.modifiers.shift)) {
                    self.segmenter.pan += response.drag_delta();
                }

                let img_w = base_tex.size()[0] as f32 * self.segmenter.scale;
                let img_h = base_tex.size()[1] as f32 * self.segmenter.scale;
                let center = rect.center() + self.segmenter.pan;
                let image_rect = Rect::from_center_size(center, [img_w, img_h].into());

                let painter = ui.painter().with_clip_rect(rect);
                painter.image(base_tex.id(), image_rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                painter.image(mask_tex.id(), image_rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);

                if let Some(pos) = response.hover_pos() {
                    if rect.contains(pos) {
                        painter.circle_stroke(pos, self.segmenter.brush_size / 2.0, Stroke::new(1.0, Color32::WHITE));
                    }
                }

                // Drawing Logic
                let active_draw = (response.dragged_by(PointerButton::Primary) || (response.hovered() && ctx.input(|i| i.pointer.primary_down()))) && !ctx.input(|i| i.modifiers.shift);

                if active_draw {
                    let pointer_pos = response.interact_pointer_pos().or(response.hover_pos());
                    if let Some(pointer_pos) = pointer_pos {
                        let rel_x = (pointer_pos.x - image_rect.min.x) / image_rect.width();
                        let rel_y = (pointer_pos.y - image_rect.min.y) / image_rect.height();

                        if rel_x >= 0.0 && rel_x <= 1.0 && rel_y >= 0.0 && rel_y <= 1.0 {
                            if let Some(mask) = &mut self.segmenter.mask_image {
                                if self.segmenter.active_brush_index < self.config.brushes.len() {
                                    let color = self.config.brushes[self.segmenter.active_brush_index].color;
                                    
                                    let w = mask.width() as f32;
                                    let h = mask.height() as f32;
                                    let cx = rel_x * w;
                                    let cy = rel_y * h;
                                    let r = (self.segmenter.brush_size / image_rect.width()) * w / 2.0;
                                    let r_sq = r * r;
                                    
                                    let min_x = (cx - r).max(0.0) as u32;
                                    let max_x = (cx + r).min(w - 1.0) as u32;
                                    let min_y = (cy - r).max(0.0) as u32;
                                    let max_y = (cy + r).min(h - 1.0) as u32;

                                    for y in min_y..=max_y {
                                        for x in min_x..=max_x {
                                            let dx = x as f32 - cx;
                                            let dy = y as f32 - cy;
                                            if dx*dx + dy*dy <= r_sq {
                                                mask.put_pixel(x, y, Rgba(color));
                                            }
                                        }
                                    }
                                    self.update_mask_texture(ctx);
                                }
                            }
                        }
                    }
                }
            } else {
                 ui.centered_and_justified(|ui| {
                    if self.classifier.workspace_path.is_none() {
                        ui.label("Load Workspace in Tab 1.");
                    } else {
                        ui.label("Load Next Image");
                    }
                });
            }
        });
    }
}