#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use eframe::{
    egui::{self, Color32, Key, Layout, PointerButton, Pos2, Rect, RichText, Sense, Stroke, TextureOptions, Vec2},
    epaint::{ColorImage, TextureHandle},
};
use image::{DynamicImage, Rgba, RgbaImage};
use rand::seq::SliceRandom;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

// --- Constants ---
const COLOR_HOLE: [u8; 4] = [227, 26, 28, 255];
const COLOR_MINING: [u8; 4] = [254, 13, 240, 255];
const COLOR_SKELETONIZER: [u8; 4] = [106, 61, 154, 255];
const COLOR_SURFACE: [u8; 4] = [218, 137, 55, 255];
const COLOR_ERASER: [u8; 4] = [0, 0, 0, 0];

#[derive(PartialEq)]
enum AppTab {
    Classifier,
    Segmentation,
}

#[derive(Clone, Copy, PartialEq)]
enum BrushType {
    Hole,
    Mining,
    Skeletonizer,
    Surface,
    Eraser,
}

impl BrushType {
    fn to_rgba(&self) -> [u8; 4] {
        match self {
            BrushType::Hole => COLOR_HOLE,
            BrushType::Mining => COLOR_MINING,
            BrushType::Skeletonizer => COLOR_SKELETONIZER,
            BrushType::Surface => COLOR_SURFACE,
            BrushType::Eraser => COLOR_ERASER,
        }
    }

    fn color32(&self) -> Color32 {
        let [r, g, b, a] = self.to_rgba();
        if a == 0 {
            Color32::WHITE
        } else {
            Color32::from_rgb(r, g, b)
        }
    }
}

// --- Structs for State Management ---

struct ClassificationAction {
    original_source: PathBuf,
    copied_destination: PathBuf,
    label: String,
}

struct ClassifierState {
    workspace_path: Option<PathBuf>,
    
    current_image_path: Option<PathBuf>,
    current_texture: Option<TextureHandle>,

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

    brush_type: BrushType,
    brush_size: f32,
    
    pan: Vec2,
    scale: f32,
}

struct MyApp {
    current_tab: AppTab,
    classifier: ClassifierState,
    segmenter: SegmentationState,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            current_tab: AppTab::Classifier,
            classifier: ClassifierState {
                workspace_path: None,
                current_image_path: None,
                current_texture: None,
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
                brush_type: BrushType::Hole,
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
            .with_inner_size([1280.0, 800.0])
            .with_title("Image Classifier & Segmenter"),
        ..Default::default()
    };
    eframe::run_native(
        "Image Workstation",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    )
}

impl MyApp {
    // --- Shared Helpers ---
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

    // --- Tab 1 Logic ---

    fn load_random_classifier_image(&mut self, ctx: &egui::Context) {
        if let Some(ws) = &self.classifier.workspace_path {
            let all_files: Vec<PathBuf> = fs::read_dir(ws)
                .ok()
                .map(|iter| {
                    iter.filter_map(|entry| entry.ok())
                        .map(|e| e.path())
                        .filter(|p| p.is_file() && is_image(p))
                        .collect()
                })
                .unwrap_or_default();

            let candidates: Vec<&PathBuf> = all_files
                .iter()
                .filter(|p| !self.classifier.processed_session_files.contains(*p))
                .collect();

            if let Some(random_file) = candidates.choose(&mut rand::thread_rng()) {
                match Self::load_image_to_texture(ctx, random_file) {
                    Ok((tex, _)) => {
                        self.classifier.current_texture = Some(tex);
                        self.classifier.current_image_path = Some((*random_file).clone());
                        self.classifier.status_msg =
                            format!("Loaded: {:?}", random_file.file_name().unwrap());
                    }
                    Err(e) => self.classifier.status_msg = format!("Error loading image: {}", e),
                }
            } else {
                self.classifier.current_texture = None;
                self.classifier.current_image_path = None;
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

            // COPY logic
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
                    self.classifier.current_image_path = Some(action.original_source);
                    self.classifier.status_msg = format!("Undid: {}", action.label);
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

    // --- Tab 2 Logic ---

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
                        .filter(|p| p.is_file() && is_image(p))
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

fn is_image(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy().to_lowercase();
        return matches!(ext_str.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "tiff");
    }
    false
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Shortcuts ---
        if self.current_tab == AppTab::Segmentation {
            if ctx.input(|i| i.key_pressed(Key::Plus) || i.key_pressed(Key::Equals)) {
                self.segmenter.brush_size += 2.0;
            }
            if ctx.input(|i| i.key_pressed(Key::Minus)) {
                self.segmenter.brush_size = (self.segmenter.brush_size - 2.0).max(1.0);
            }
        }

        // --- Top Menu (Always on top) ---
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut self.current_tab,
                    AppTab::Classifier,
                    " 1. Classifier ",
                );
                if ui
                    .selectable_value(
                        &mut self.current_tab,
                        AppTab::Segmentation,
                        " 2. Segmentation ",
                    )
                    .clicked()
                {
                    if self.segmenter.images_queue.is_empty()
                        && self.segmenter.base_texture.is_none()
                    {
                        self.init_segmentation_queue();
                        self.load_next_segmentation(ctx);
                    }
                }
                
                ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.current_tab == AppTab::Classifier {
                        ui.label(RichText::new(format!("Session Classified: {}", self.classifier.session_count)).strong());
                    } else {
                        ui.label(format!("Remaining: {}", self.segmenter.images_queue.len()));
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
    fn ui_classifier(&mut self, ctx: &egui::Context) {
        // --- 1. Bottom Button Panel (Anchored Bottom) ---
        // FIX: Use exact_height to prevent layout collapse
        egui::TopBottomPanel::bottom("class_bottom_panel")
            .resizable(false)
            .exact_height(100.0) // Force height to 100px
            .show(ctx, |ui| {
                // Use a centered horizontal layout inside the fixed height panel
                ui.with_layout(egui::Layout::centered_and_justified(egui::Direction::LeftToRight), |ui| {
                     ui.horizontal(|ui| {
                        let btn_h = 50.0;
                        let btn_w = 140.0;
                        
                        let has_image = self.classifier.current_texture.is_some();

                        let big_btn = |ui: &mut egui::Ui, text: &str, color: Color32| {
                            ui.add_enabled(
                                has_image,
                                egui::Button::new(RichText::new(text).size(18.0).strong().color(Color32::WHITE))
                                    .min_size(Vec2::new(btn_w, btn_h))
                                    .fill(color),
                            )
                        };

                        if big_btn(ui, "Healthy", Color32::from_rgb(46, 125, 50)).clicked() {
                            self.classify_current(ctx, "Healthy");
                        }
                        ui.add_space(15.0);
                        if big_btn(ui, "Undecided", Color32::from_rgb(117, 117, 117)).clicked() {
                            self.classify_current(ctx, "Undecided");
                        }
                        ui.add_space(15.0);
                        if big_btn(ui, "Damaged", Color32::from_rgb(198, 40, 40)).clicked() {
                            self.classify_current(ctx, "Damaged");
                        }

                        ui.add_space(40.0);
                        ui.separator();
                        ui.add_space(40.0);

                        let undo_enabled = !self.classifier.undo_stack.is_empty();
                        let undo_btn = egui::Button::new(RichText::new("Undo").size(18.0))
                            .min_size(Vec2::new(btn_w, btn_h));
                        
                        if ui.add_enabled(undo_enabled, undo_btn).clicked() {
                            self.undo_last_classification(ctx);
                        }
                    });
                });
            });

        // --- 2. Left Sidebar (With ScrollArea) ---
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
                            self.load_random_classifier_image(ctx);
                        }
                    }
                    
                    ui.label(RichText::new(&self.classifier.status_msg).size(12.0).italics());

                    ui.separator();
                    ui.heading("Last Classified");
                    ui.add_space(10.0);

                    if let Some(tex) = &self.classifier.last_classified_texture {
                        ui.label(RichText::new(&self.classifier.last_classified_label).strong().size(16.0));
                        ui.add(egui::Image::new(tex).max_width(200.0));
                    } else {
                        ui.label("No history yet.");
                    }
                });
            });

        // --- 3. Central Panel (Fills remaining space) ---
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(tex) = &self.classifier.current_texture {
                let avail_size = ui.available_size();
                ui.centered_and_justified(|ui| {
                    ui.add(egui::Image::new(tex).max_size(avail_size));
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.heading("No Image Loaded");
                });
            }
        });
    }

    fn ui_segmentation(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("seg_tools_panel").show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Tools");
                ui.separator();

                let mut brush_btn = |b_type: BrushType, label: &str| {
                    let is_selected = self.segmenter.brush_type == b_type;
                    let color = if b_type == BrushType::Eraser {
                        Color32::WHITE
                    } else {
                        b_type.color32()
                    };
                    let text = RichText::new(label).color(color).strong();
                    if ui.selectable_label(is_selected, text).clicked() {
                        self.segmenter.brush_type = b_type;
                    }
                };

                brush_btn(BrushType::Eraser, "Eraser");
                brush_btn(BrushType::Hole, "Hole (Red)");
                brush_btn(BrushType::Mining, "Mining (Pink)");
                brush_btn(BrushType::Skeletonizer, "Skeleton (Purple)");
                brush_btn(BrushType::Surface, "Surface (Orange)");

                ui.separator();
                ui.label(format!("Brush Size: {:.0}", self.segmenter.brush_size));
                ui.add(egui::Slider::new(&mut self.segmenter.brush_size, 1.0..=100.0));

                ui.add_space(20.0);
                if ui.button("Reset Mask").clicked() {
                    if let Some(mask) = &mut self.segmenter.mask_image {
                        *mask = RgbaImage::new(mask.width(), mask.height());
                        self.update_mask_texture(ctx);
                    }
                }

                ui.add_space(20.0);
                ui.separator();
                
                let btn_size = Vec2::new(120.0, 30.0);
                if ui.add(egui::Button::new("Save").min_size(btn_size)).clicked() {
                    self.save_mask();
                }
                if ui.add(egui::Button::new("Save & Next").min_size(btn_size)).clicked() {
                    self.save_mask();
                    self.load_next_segmentation(ctx);
                }
                if ui.add(egui::Button::new("Skip / Next").min_size(btn_size)).clicked() {
                    self.load_next_segmentation(ctx);
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let (Some(base_tex), Some(mask_tex)) = (
                &self.segmenter.base_texture,
                &self.segmenter.mask_texture,
            ) {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());

                // Zoom control
                let scroll = ctx.input(|i| i.raw_scroll_delta);
                if scroll.y != 0.0 {
                    let zoom_factor = if scroll.y > 0.0 { 1.1 } else { 0.9 };
                    self.segmenter.scale *= zoom_factor;
                }

                // Pan control (Middle Mouse or Shift+Drag)
                if response.dragged_by(PointerButton::Middle)
                    || (response.dragged() && ctx.input(|i| i.modifiers.shift))
                {
                    self.segmenter.pan += response.drag_delta();
                }

                let img_w = base_tex.size()[0] as f32 * self.segmenter.scale;
                let img_h = base_tex.size()[1] as f32 * self.segmenter.scale;
                let center = rect.center() + self.segmenter.pan;
                let image_rect = Rect::from_center_size(center, [img_w, img_h].into());

                let painter = ui.painter().with_clip_rect(rect);

                // Layers
                painter.image(
                    base_tex.id(),
                    image_rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
                painter.image(
                    mask_tex.id(),
                    image_rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );

                // Brush Outline
                if let Some(mouse_pos) = response.hover_pos() {
                    if rect.contains(mouse_pos) {
                         painter.circle_stroke(
                            mouse_pos,
                            self.segmenter.brush_size / 2.0,
                            Stroke::new(1.0, Color32::WHITE),
                        );
                    }
                }

                // Drawing
                if response.dragged_by(PointerButton::Primary) && !ctx.input(|i| i.modifiers.shift) {
                    if let Some(pointer_pos) = response.interact_pointer_pos() {
                        let rel_x = (pointer_pos.x - image_rect.min.x) / image_rect.width();
                        let rel_y = (pointer_pos.y - image_rect.min.y) / image_rect.height();

                        if rel_x >= 0.0 && rel_x <= 1.0 && rel_y >= 0.0 && rel_y <= 1.0 {
                            if let Some(mask) = &mut self.segmenter.mask_image {
                                let w = mask.width() as f32;
                                let h = mask.height() as f32;
                                let cx = rel_x * w;
                                let cy = rel_y * h;

                                let r = (self.segmenter.brush_size / image_rect.width()) * w / 2.0;
                                let color = self.segmenter.brush_type.to_rgba();
                                let r_sq = r * r;

                                let min_x = (cx - r).max(0.0) as u32;
                                let max_x = (cx + r).min(w - 1.0) as u32;
                                let min_y = (cy - r).max(0.0) as u32;
                                let max_y = (cy + r).min(h - 1.0) as u32;

                                for y in min_y..=max_y {
                                    for x in min_x..=max_x {
                                        let dx = x as f32 - cx;
                                        let dy = y as f32 - cy;
                                        if dx * dx + dy * dy <= r_sq {
                                            mask.put_pixel(x, y, Rgba(color));
                                        }
                                    }
                                }
                                self.update_mask_texture(ctx);
                            }
                        }
                    }
                }
            } else {
                 ui.centered_and_justified(|ui| {
                    if self.classifier.workspace_path.is_none() {
                        ui.label("Please load workspace in Tab 1 first.");
                    } else if self.segmenter.images_queue.is_empty() {
                         ui.label("No pending 'Damaged' images found (checked against masks folder).");
                    } else {
                        ui.label("Load Next Image");
                    }
                });
            }
        });
    }
}