
import { Injectable, computed, signal } from '@angular/core';
import {
  STYLE_PRESETS, CAMERA_PRESETS, CAMERA_DISTANCES,
  StylePreset, CameraPreset, CameraDistancePreset,
} from '../models/style-presets';
import { Character, CHARACTER_SLIDER_DEFS } from '../models/character.model';
import { Location, LOCATION_SLIDER_DEFS } from '../models/location.model';

@Injectable({ providedIn: 'root' })
export class StylePresetService {
  /** Active style ID (mutually exclusive — at most 1) */
  readonly activeStyleId = signal<string | null>(null);

  /** Selected camera angle preset (null = no camera direction) */
  readonly selectedCameraPreset = signal<CameraPreset | null>(null);

  /** Selected camera distance (null = default/medium) */
  readonly selectedCameraDistance = signal<CameraDistancePreset | null>(null);

  /** Bound character data from scene */
  private readonly boundCharacter = signal<Character | null>(null);

  /** Bound location data from scene */
  private readonly boundLocation = signal<Location | null>(null);

  /** Character slider values (from bound character's sliders) */
  readonly characterSliderValues = computed(() => this.boundCharacter()?.sliders ?? {});

  /** Location slider values (from bound location's sliders) */
  readonly locationSliderValues = computed(() => this.boundLocation()?.sliders ?? {});

  /** Bound character (read-only for template) */
  readonly character = computed(() => this.boundCharacter());

  /** Bound location (read-only for template) */
  readonly location = computed(() => this.boundLocation());

  /** Data */
  readonly styles = STYLE_PRESETS;
  readonly cameraPresets = CAMERA_PRESETS;
  readonly cameraDistances = CAMERA_DISTANCES;

  /** Resolved active style preset */
  readonly activeStyle = computed(() => {
    const id = this.activeStyleId();
    return id ? STYLE_PRESETS.find(p => p.id === id) ?? null : null;
  });

  // ── Style (mutually exclusive) ────────────────────────

  toggleStyle(presetId: string) {
    this.activeStyleId.update(cur => cur === presetId ? null : presetId);
  }

  // ── Scene Binding ──────────────────────────────────────

  setCharacterData(char: Character | null) {
    this.boundCharacter.set(char);
  }

  setLocationData(loc: Location | null) {
    this.boundLocation.set(loc);
  }

  // ── Camera ────────────────────────────────────────────

  selectCameraPosition(preset: CameraPreset | null) {
    this.selectedCameraPreset.set(preset);
  }

  selectCameraDistance(dist: CameraDistancePreset | null) {
    this.selectedCameraDistance.set(dist);
  }

  resetCamera() {
    this.selectedCameraPreset.set(null);
    this.selectedCameraDistance.set(null);
  }

  // ── Prompt Building ───────────────────────────────────

  buildFinalPrompt(userPrompt: string): string {
    const parts: string[] = [];

    const style = this.activeStyle();
    if (style?.promptKeywords) {
      parts.push(style.promptKeywords);
    }

    // Character prompt
    const char = this.boundCharacter();
    if (char?.prompt) {
      parts.push(char.prompt);
    }

    // Location prompt
    const loc = this.boundLocation();
    if (loc?.prompt) {
      parts.push(loc.prompt);
    }

    // Camera angle prompts
    const cam = this.selectedCameraPreset();
    if (cam) {
      parts.push(...cam.prompts.filter(p => p));
    }

    // Camera distance prompt
    const dist = this.selectedCameraDistance();
    if (dist?.prompt) {
      parts.push(dist.prompt);
    }

    if (userPrompt.trim()) {
      parts.push(userPrompt.trim());
    }

    return parts.join(', ');
  }

  // ── LoRA Merging ──────────────────────────────────────

  mergeLoRAs(manualLoRAs: { filename: string; scale: number }[]): { filename: string; scale: number }[] {
    const merged = new Map<string, number>();

    // Style LoRAs (fixed scale from preset definition)
    const style = this.activeStyle();
    if (style) {
      for (const lora of style.loras) {
        merged.set(lora.filename, lora.scale);
      }
    }

    // Camera angle LoRAs
    const cam = this.selectedCameraPreset();
    if (cam) {
      for (const lora of cam.loras) {
        merged.set(lora.filename, lora.scale);
      }
    }

    // Camera distance LoRA
    const dist = this.selectedCameraDistance();
    if (dist?.lora) {
      merged.set(dist.lora.filename, dist.lora.scale);
    }

    // Character slider LoRAs
    const charSliders = this.characterSliderValues();
    for (const def of CHARACTER_SLIDER_DEFS) {
      const val = charSliders[def.id] ?? 0;
      if (val !== 0) {
        merged.set(def.loraFilename, val);
      }
    }

    // Location slider LoRAs
    const locSliders = this.locationSliderValues();
    for (const def of LOCATION_SLIDER_DEFS) {
      const val = locSliders[def.id] ?? 0;
      if (val !== 0) {
        merged.set(def.loraFilename, val);
      }
    }

    // Manual LoRAs override if same filename
    for (const lora of manualLoRAs) {
      merged.set(lora.filename, lora.scale);
    }

    return Array.from(merged.entries()).map(([filename, scale]) => ({ filename, scale }));
  }
}
