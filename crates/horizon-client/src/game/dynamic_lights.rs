//! Horizon Extras' dynamic lights (`horizon_client::dynamic_light`): the light sources near the
//! camera this frame. Nothing is gathered while the extra is off.
use super::{Game, shield};
use glam::DVec3;
use horizon_client::dynamic_light::{self, BURNING, Luminance, Source};

impl Game<'_> {
    /// Fills `dynamic_lights` for the frame: the player's and mobs' held items and fire, and
    /// glowing dropped items, the nearest [`dynamic_light::MAX`] of them.
    pub(super) fn gather_dynamic_lights(&mut self, camera: DVec3, alpha: f64) {
        let mut out = std::mem::take(&mut self.dynamic_lights);
        let mut colors = std::mem::take(&mut self.dynamic_light_colors);
        out.clear();
        colors.clear();
        if !self.settings.extras.dynamic_lights {
            self.dynamic_lights = out;
            self.dynamic_light_colors = colors;
            return;
        }
        if self.luminance.is_none() {
            self.luminance = Some(Luminance::new(self.res.facts(), self.pack.assets.registry));
        }
        let lum = self.luminance.as_ref().expect("built above");
        let held = |item: Option<&str>| item.map_or(0, |i| lum.of(i));
        let tint = |item: Option<&str>, burning: bool| -> [f32; 3] {
            if burning {
                [1.0, 0.28, 0.06]
            } else {
                let name = item.unwrap_or("");
                if name.contains("soul_torch") || name.contains("soul_lantern") || name.contains("soul_fire") {
                    [0.20, 0.62, 1.0]
                } else if name.contains("redstone_torch") {
                    [1.0, 0.12, 0.06]
                } else if name.contains("lava") {
                    [1.0, 0.24, 0.04]
                } else if name.contains("sea_lantern") || name.contains("prismarine") {
                    [0.30, 0.85, 1.0]
                } else if name.contains("end_rod") {
                    [0.78, 0.65, 1.0]
                } else {
                    [1.0, 0.68, 0.32]
                }
            }
        };
        let mut sources = Vec::new();
        if let Some(p) = &self.player
            && !p.spectator
        {
            let feet = self
                .previous_position
                .lerp(p.position, self.clock.camera_fraction());
            let level = held(self.stack_in(self.selected).map(|s| &*s.item))
                .max(held(self.stack_in(shield::OFFHAND_SLOT).map(|s| &*s.item)))
                .max(if p.fire_ticks > 0 { BURNING } else { 0 });
            // About where the hands are.
            if level > 0 {
                let item = self.stack_in(self.selected).map(|s| &*s.item);
                sources.push(Source {
                    at: DVec3::new(feet.x, feet.y + 1.2, feet.z),
                    level,
                    color: tint(item, p.fire_ticks > 0),
                });
            }
        }
        for m in self.mobs.iter().filter(|m| !m.removed) {
            let level = held(m.extra.main_hand.as_deref())
                .max(held(m.extra.off_hand.as_deref()))
                .max(if m.fire_ticks > 0 { BURNING } else { 0 });
            if level == 0 {
                continue;
            }
            let at = m.previous.lerp(m.position, alpha);
            let item = m.extra.main_hand.as_deref().or(m.extra.off_hand.as_deref());
            sources.push(Source {
                at: DVec3::new(at.x, at.y + f64::from(m.height()) * 0.6, at.z),
                level,
                color: tint(item, m.fire_ticks > 0),
            });
        }
        for item in &self.dropped {
            let level = lum
                .of(&item.stack.item)
                .max(if item.fire_ticks > 0 { BURNING } else { 0 });
            if level == 0 {
                continue;
            }
            let at = item.previous.lerp(item.position, alpha);
            sources.push(Source {
                at: DVec3::new(at.x, at.y + 0.25, at.z),
                level,
                color: tint(Some(&item.stack.item), item.fire_ticks > 0),
            });
        }
        dynamic_light::select(&sources, camera, &mut out, &mut colors);
        self.dynamic_lights = out;
        self.dynamic_light_colors = colors;
    }
}
