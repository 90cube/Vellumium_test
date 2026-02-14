
import { Component, inject, signal, effect, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { CanvasService } from '../../services/canvas.service';
import { VellumiumService } from '../../services/vellumium.service';
import { StylePresetService } from '../../services/style-preset.service';
import { CharacterService } from '../../services/character.service';
import { LocationService } from '../../services/location.service';
import {
  GenerationApiService, GenerationMode, GenerationParams, ControlNetType,
  LoRASelection, SCHEDULERS, QueueItem
} from '../../services/generation-api.service';
import { STYLE_PRESETS, STYLE_GROUP_LABELS, StylePreset, StyleGroup } from '../../models/style-presets';
import { CameraPickerComponent } from './camera-picker.component';

@Component({
  selector: 'app-canvas-generation-panel',
  standalone: true,
  imports: [CommonModule, FormsModule, CameraPickerComponent],
  template: `
    <div class="flex flex-col h-full">

      <!-- Mode Tabs -->
      <div class="flex border-b border-[--cinema-border] shrink-0">
        @for (tab of modeTabs; track tab.id) {
          <button
            class="flex-1 py-2.5 text-[10px] font-display font-semibold uppercase tracking-wider transition-all"
            [class]="activeMode() === tab.id
              ? 'text-[--cinema-primary] border-b-2 border-[--cinema-primary]'
              : 'text-[--cinema-text-muted] hover:text-[--cinema-primary]'"
            (click)="activeMode.set(tab.id)">{{ tab.label }}</button>
        }
      </div>

      <div class="flex-1 overflow-y-auto p-4 space-y-4">

        <!-- Scene Bindings (read-only summary) -->
        @if (activeMode() !== 'qwen_edit') {
          <div class="space-y-1.5">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Scene Bindings</label>
            <div class="space-y-1">
              <div class="flex items-center gap-2 px-2 py-1.5 rounded border border-[--cinema-border] bg-[--cinema-bg]">
                <span class="material-symbols-outlined text-[14px] text-[--cinema-text-muted]">person</span>
                <span class="text-[10px]"
                  [class]="presetService.character()
                    ? 'text-[--cinema-text]'
                    : 'text-[--cinema-text-dim]'">{{ presetService.character()?.name ?? 'No character bound' }}</span>
              </div>
              <div class="flex items-center gap-2 px-2 py-1.5 rounded border border-[--cinema-border] bg-[--cinema-bg]">
                <span class="material-symbols-outlined text-[14px] text-[--cinema-text-muted]">location_on</span>
                <span class="text-[10px]"
                  [class]="presetService.location()
                    ? 'text-[--cinema-text]'
                    : 'text-[--cinema-text-dim]'">{{ presetService.location()?.name ?? 'No location bound' }}</span>
              </div>
            </div>
          </div>
        }

        <!-- Style (dropdown selector) -->
        @if (activeMode() !== 'qwen_edit') {
          <div class="space-y-1.5 relative">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Style</label>
            <button
              (click)="styleDropdownOpen.set(!styleDropdownOpen())"
              class="w-full flex items-center justify-between px-3 py-1.5 bg-[--cinema-bg] border border-[--cinema-border] rounded-lg text-xs transition-colors hover:border-[--cinema-primary]/30">
              @if (presetService.activeStyle(); as style) {
                <span class="text-[--cinema-text]">{{ style.label }}</span>
              } @else {
                <span class="text-[--cinema-text-dim]">Select style...</span>
              }
              <span class="material-symbols-outlined text-[14px] text-[--cinema-text-muted] transition-transform"
                [class.rotate-180]="styleDropdownOpen()">expand_more</span>
            </button>
            @if (styleDropdownOpen()) {
              <div class="absolute left-0 right-0 top-full mt-1 z-50 bg-[--cinema-surface] border border-[--cinema-border] rounded-lg shadow-xl max-h-[240px] overflow-y-auto">
                <button
                  (click)="presetService.toggleStyle(''); styleDropdownOpen.set(false)"
                  class="w-full text-left px-3 py-1.5 text-[10px] text-[--cinema-text-dim] hover:bg-white/5">None</button>
                @for (group of styleGroups; track group.key) {
                  <div class="border-t border-[--cinema-border]">
                    <div class="px-3 py-1 text-[9px] font-display font-semibold text-[--cinema-text-dim] uppercase tracking-wider">{{ group.label }}</div>
                    @for (style of group.items; track style.id) {
                      <button
                        (click)="presetService.toggleStyle(style.id); styleDropdownOpen.set(false)"
                        class="w-full text-left px-3 py-1.5 text-[10px] flex items-center gap-2 transition-colors"
                        [class]="presetService.activeStyleId() === style.id
                          ? 'bg-[--cinema-primary]/10 text-[--cinema-primary]'
                          : 'text-[--cinema-text] hover:bg-white/5'">
                        @if (style.loras.length > 0) {
                          <span class="w-1.5 h-1.5 rounded-full bg-[--cinema-accent]"></span>
                        } @else {
                          <span class="w-1.5 h-1.5 rounded-full bg-[--cinema-text-dim]/30"></span>
                        }
                        {{ style.label }}
                        @if (presetService.activeStyleId() === style.id) {
                          <span class="material-symbols-outlined text-[12px] ml-auto">check</span>
                        }
                      </button>
                    }
                  </div>
                }
              </div>
            }
          </div>
        }

        <!-- Prompt -->
        @if (activeMode() !== 'qwen_edit') {
          <div class="space-y-1.5">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider flex items-center justify-between">
              Prompt
              <span class="material-symbols-outlined text-[14px] text-[--cinema-primary]/40">auto_awesome</span>
            </label>
            <textarea
              [(ngModel)]="prompt"
              class="w-full h-20 bg-[--cinema-bg] text-[--cinema-text] text-xs border border-[--cinema-border] rounded-lg px-3 py-2 focus:ring-1 focus:ring-[--cinema-primary]/30 focus:border-[--cinema-primary]/50 focus:outline-none resize-none"
              placeholder="Describe the image..."></textarea>
          </div>
        }

        <!-- 3D Camera Picker -->
        @if (activeMode() !== 'qwen_edit') {
          <app-camera-picker></app-camera-picker>
        }

        <!-- Qwen Edit mode specific -->
        @if (activeMode() === 'qwen_edit') {
          <div class="space-y-1.5">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Edit Instruction</label>
            <textarea
              [(ngModel)]="qwenInstruction"
              class="w-full h-16 bg-[--cinema-bg] text-[--cinema-text] text-xs border border-[--cinema-border] rounded-lg px-3 py-2 focus:ring-1 focus:ring-[--cinema-primary]/30 focus:border-[--cinema-primary]/50 focus:outline-none resize-none"
              placeholder="Describe what to change..."></textarea>
          </div>

          <div class="space-y-1.5">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Source Images (from layers)</label>
            <div class="flex flex-wrap gap-1">
              @for (layer of canvasService.layers(); track layer.id) {
                @if (layer.type === 'image') {
                  <button
                    class="px-2 py-1 rounded text-[10px] border transition-colors"
                    [class]="qwenSourceLayers.has(layer.id)
                      ? 'bg-[--cinema-primary]/15 text-[--cinema-primary] border-[--cinema-primary]/30'
                      : 'bg-[--cinema-panel] text-[--cinema-text-muted] border-[--cinema-border] hover:border-[--cinema-primary]/30'"
                    (click)="toggleQwenSource(layer.id)">
                    {{ layer.name }}
                  </button>
                }
              }
            </div>
            <span class="text-[10px] text-[--cinema-text-dim]">Select 1-3 image layers</span>
          </div>

          <div class="space-y-1.5">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Reference Images (optional)</label>
            @if (qwenRefImagesBase64.length > 0) {
              <div class="flex gap-2 flex-wrap">
                @for (img of qwenRefImagesBase64; track $index) {
                  <div class="relative group">
                    <img [src]="img" class="w-16 h-16 object-cover rounded-lg border border-[--cinema-border]">
                    <button
                      class="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-red-500/80 text-white flex items-center justify-center text-[10px] leading-none opacity-0 group-hover:opacity-100 transition-opacity"
                      (click)="removeRefImage($index)">x</button>
                  </div>
                }
              </div>
            }
            <input type="file" accept="image/*" multiple
              class="w-full text-[10px] text-[--cinema-text-muted] file:mr-2 file:py-1 file:px-2 file:rounded file:border file:border-[--cinema-border] file:bg-[--cinema-panel] file:text-[--cinema-text-muted] file:text-[10px] file:cursor-pointer"
              (change)="onRefImagesSelected($event)">
          </div>
        }

        <!-- Frame Ratio + Resolution -->
        @if (activeMode() !== 'qwen_edit') {
          <div class="space-y-1.5">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider flex items-center justify-between">
              Frame Ratio
              <div class="flex gap-1">
                <button class="px-1.5 py-0.5 rounded text-[9px] border transition-colors"
                  [class]="resolutionTier() === '1MP' ? 'bg-[--cinema-primary]/15 text-[--cinema-primary] border-[--cinema-primary]/30' : 'text-[--cinema-text-dim] border-[--cinema-border]'"
                  (click)="resolutionTier.set('1MP')">1MP</button>
                <button class="px-1.5 py-0.5 rounded text-[9px] border transition-colors"
                  [class]="resolutionTier() === '2MP' ? 'bg-[--cinema-primary]/15 text-[--cinema-primary] border-[--cinema-primary]/30' : 'text-[--cinema-text-dim] border-[--cinema-border]'"
                  (click)="resolutionTier.set('2MP')">2MP</button>
              </div>
            </label>
            <div class="flex flex-wrap gap-1">
              @for (r of canvasService.FRAME_RATIOS; track r.value) {
                <button class="px-2 py-1 text-[10px] rounded border transition-colors"
                  [class]="canvasService.frameAspectRatio() === r.value
                    ? 'bg-[--cinema-primary]/15 text-[--cinema-primary] border-[--cinema-primary]/30'
                    : 'text-[--cinema-text-muted] border-[--cinema-border] hover:border-[--cinema-primary]/30'"
                  (click)="canvasService.setFrameAspectRatio(r.value)">{{ r.label }}</button>
              }
            </div>
            <div class="text-[10px] text-[--cinema-text-dim]">
              Generation: {{ getGenerationResolution().w }} x {{ getGenerationResolution().h }}
            </div>
          </div>
        }

        <!-- Steps & Seed -->
        @if (activeMode() !== 'qwen_edit') {
          <div class="grid grid-cols-2 gap-3">
            <div class="space-y-1">
              <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Steps</label>
              <input type="number" [(ngModel)]="steps" min="1" max="100"
                class="w-full bg-[--cinema-bg] text-[--cinema-text] text-xs border border-[--cinema-border] rounded-lg px-3 py-1.5 outline-none focus:border-[--cinema-primary]/30">
            </div>
            <div class="space-y-1">
              <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider flex items-center justify-between">
                Seed
                <button class="text-[--cinema-text-dim] hover:text-[--cinema-primary]"
                  (click)="seed = -1" title="Random">
                  <span class="material-symbols-outlined text-[12px]">casino</span>
                </button>
              </label>
              <input type="number" [(ngModel)]="seed" min="-1"
                class="w-full bg-[--cinema-bg] text-[--cinema-text] text-xs border border-[--cinema-border] rounded-lg px-3 py-1.5 outline-none focus:border-[--cinema-primary]/30">
            </div>
          </div>
        }

        <!-- Scheduler -->
        @if (activeMode() !== 'qwen_edit') {
          <div class="space-y-1.5">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Scheduler</label>
            <select
              class="w-full bg-[--cinema-bg] text-[--cinema-text] text-xs border border-[--cinema-border] rounded-lg px-3 py-1.5 outline-none focus:border-[--cinema-primary]/30"
              [(ngModel)]="scheduler">
              @for (s of schedulers; track s) {
                <option [value]="s">{{ s }}</option>
              }
            </select>
          </div>
        }

        <!-- Inpaint specific -->
        @if (activeMode() === 'inpaint') {
          <div class="space-y-3 p-3 bg-[--cinema-bg] rounded-lg border border-[--cinema-border]">
            <span class="text-[10px] font-display font-semibold text-[--cinema-primary] uppercase tracking-wider">Inpaint Settings</span>

            <div class="space-y-1.5">
              <div class="flex items-center justify-between">
                <label class="text-[10px] text-[--cinema-text-muted]">Mask Blur</label>
                <span class="text-[10px] text-[--cinema-text]">{{ maskBlur }}</span>
              </div>
              <input type="range" min="0" max="64" [(ngModel)]="maskBlur"
                class="w-full h-1 accent-[--cinema-primary]">
            </div>

            <div class="space-y-1.5">
              <div class="flex items-center justify-between">
                <label class="text-[10px] text-[--cinema-text-muted]">Denoise Strength</label>
                <span class="text-[10px] text-[--cinema-text]">{{ denoiseStrength.toFixed(2) }}</span>
              </div>
              <input type="range" min="10" max="100" [value]="denoiseStrength * 100"
                (input)="denoiseStrength = +$any($event.target).value / 100"
                class="w-full h-1 accent-[--cinema-primary]">
            </div>

            <div class="space-y-1">
              <label class="text-[10px] text-[--cinema-text-muted]">Mask Source</label>
              <div class="flex gap-1">
                <button class="flex-1 py-1 text-[10px] rounded border transition-colors"
                  [class]="maskSource === 'canvas' ? 'bg-[--cinema-primary]/15 text-[--cinema-primary] border-[--cinema-primary]/30' : 'text-[--cinema-text-muted] border-[--cinema-border]'"
                  (click)="maskSource = 'canvas'">Canvas Mask</button>
                <button class="flex-1 py-1 text-[10px] rounded border transition-colors"
                  [class]="maskSource === 'upload' ? 'bg-[--cinema-primary]/15 text-[--cinema-primary] border-[--cinema-primary]/30' : 'text-[--cinema-text-muted] border-[--cinema-border]'"
                  (click)="maskSource = 'upload'">Upload</button>
              </div>
            </div>

            @if (maskSource === 'upload') {
              <input type="file" accept="image/*"
                class="w-full text-[10px] text-[--cinema-text-muted] file:mr-2 file:py-1 file:px-2 file:rounded file:border file:border-[--cinema-border] file:bg-[--cinema-panel] file:text-[--cinema-text-muted] file:text-[10px]"
                (change)="onMaskFileSelected($event)">
            }
          </div>
        }

        <!-- ControlNet specific -->
        @if (activeMode() === 'controlnet') {
          <div class="space-y-3 p-3 bg-[--cinema-bg] rounded-lg border border-[--cinema-border]">
            <span class="text-[10px] font-display font-semibold text-[--cinema-primary] uppercase tracking-wider">ControlNet Settings</span>

            <div class="space-y-1.5">
              <label class="text-[10px] text-[--cinema-text-muted]">Control Type</label>
              <div class="flex flex-wrap gap-1">
                @for (type of controlNetTypes; track type) {
                  <button class="px-2 py-1 text-[10px] rounded border transition-colors capitalize"
                    [class]="selectedControlType() === type ? 'bg-[--cinema-primary]/15 text-[--cinema-primary] border-[--cinema-primary]/30' : 'text-[--cinema-text-muted] border-[--cinema-border]'"
                    (click)="selectedControlType.set(type)">{{ type }}</button>
                }
              </div>
            </div>

            <div class="space-y-1.5">
              <div class="flex items-center justify-between">
                <label class="text-[10px] text-[--cinema-text-muted]">Control Scale</label>
                <span class="text-[10px] text-[--cinema-text]">{{ controlScale.toFixed(2) }}</span>
              </div>
              <input type="range" min="0" max="100" [value]="controlScale * 100"
                (input)="controlScale = +$any($event.target).value / 100"
                class="w-full h-1 accent-[--cinema-primary]">
            </div>

            <div class="space-y-1">
              <label class="text-[10px] text-[--cinema-text-muted]">Control Image</label>
              <div class="flex gap-1">
                <button class="flex-1 py-1 text-[10px] rounded border transition-colors"
                  [class]="controlImageSource === 'canvas' ? 'bg-[--cinema-primary]/15 text-[--cinema-primary] border-[--cinema-primary]/30' : 'text-[--cinema-text-muted] border-[--cinema-border]'"
                  (click)="controlImageSource = 'canvas'">From Canvas</button>
                <button class="flex-1 py-1 text-[10px] rounded border transition-colors"
                  [class]="controlImageSource === 'upload' ? 'bg-[--cinema-primary]/15 text-[--cinema-primary] border-[--cinema-primary]/30' : 'text-[--cinema-text-muted] border-[--cinema-border]'"
                  (click)="controlImageSource = 'upload'">Upload</button>
              </div>
            </div>

            @if (controlImageSource === 'upload') {
              <input type="file" accept="image/*"
                class="w-full text-[10px] text-[--cinema-text-muted] file:mr-2 file:py-1 file:px-2 file:rounded file:border file:border-[--cinema-border] file:bg-[--cinema-panel] file:text-[--cinema-text-muted] file:text-[10px]"
                (change)="onControlImageSelected($event)">
            }
          </div>
        }
      </div>

      <!-- Generate Button (sticky bottom) -->
      <div class="p-4 border-t border-[--cinema-border] space-y-2 shrink-0">
        @if (genApi.lastError()) {
          <div class="text-[--cinema-error] text-[10px] px-1">{{ genApi.lastError() }}</div>
        }
        @if (generationSuccess()) {
          <div class="text-green-400 text-[10px] px-1">Image generated successfully.</div>
        }

        <button
          (click)="generate()"
          [disabled]="genApi.isSubmitting() || !canGenerate()"
          class="w-full py-2.5 rounded-lg font-display font-semibold text-sm flex items-center justify-center gap-2 transition-all
            bg-gradient-to-r from-[--cinema-primary] to-[--cinema-accent] text-white
            hover:shadow-lg hover:shadow-[--cinema-primary]/20
            disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none">
          <span class="material-symbols-outlined text-lg">{{ activeMode() === 'qwen_edit' ? 'edit' : 'auto_awesome' }}</span>
          {{ genApi.isSubmitting() ? 'Generating...' : activeMode() === 'qwen_edit' ? 'Edit Image' : 'Generate' }}
        </button>
      </div>
    </div>
  `
})
export class CanvasGenerationPanelComponent implements OnInit, OnDestroy {
  canvasService = inject(CanvasService);
  vellumService = inject(VellumiumService);
  genApi = inject(GenerationApiService);
  presetService = inject(StylePresetService);
  private characterService = inject(CharacterService);
  private locationService = inject(LocationService);

  // Mode
  activeMode = signal<GenerationMode>('t2i');
  modeTabs: { id: GenerationMode; label: string }[] = [
    { id: 't2i', label: 'T2I' },
    { id: 'inpaint', label: 'Inpaint' },
    { id: 'controlnet', label: 'ControlNet' },
    { id: 'qwen_edit', label: 'Qwen Edit' },
  ];

  // Style dropdown
  styleDropdownOpen = signal(false);
  readonly styleGroups: { key: StyleGroup; label: string; items: StylePreset[] }[] = (() => {
    const groups: StyleGroup[] = ['pixel', 'illustration', '3d', 'film', 'character'];
    return groups
      .map(key => ({ key, label: STYLE_GROUP_LABELS[key], items: STYLE_PRESETS.filter(s => s.group === key) }))
      .filter(g => g.items.length > 0);
  })();

  // Common params
  prompt = '';
  resolutionTier = signal<'1MP' | '2MP'>('1MP');
  steps = 8;
  seed = -1;
  scheduler = SCHEDULERS[0];
  schedulers = SCHEDULERS;

  // Inpaint
  maskBlur = 8;
  denoiseStrength = 0.75;
  maskSource: 'canvas' | 'upload' = 'canvas';
  uploadedMaskBase64: string | null = null;

  // ControlNet
  controlNetTypes: ControlNetType[] = ['canny', 'hed', 'depth', 'pose', 'mlsd'];
  selectedControlType = signal<ControlNetType>('canny');
  controlScale = 0.75;
  controlImageSource: 'canvas' | 'upload' = 'canvas';
  uploadedControlBase64: string | null = null;

  // Qwen Edit
  qwenInstruction = '';
  qwenSourceLayers = new Set<string>();
  qwenRefImagesBase64: string[] = [];

  // Status
  generationSuccess = signal(false);
  private unsubQueue: (() => void) | null = null;

  constructor() {
    // Sync scene bindings to preset service whenever selected scene changes
    effect(() => {
      const scene = this.vellumService.selectedScene();
      const characters = this.characterService.characters();
      const locations = this.locationService.locations();

      const char = scene?.character_id
        ? characters.find(c => c.id === scene.character_id) ?? null
        : null;
      const loc = scene?.location_id
        ? locations.find(l => l.id === scene.location_id) ?? null
        : null;

      this.presetService.setCharacterData(char);
      this.presetService.setLocationData(loc);
    });

  }

  ngOnInit() {
    this.genApi.fetchStatus();

    const scene = this.vellumService.selectedScene();
    if (scene) {
      this.unsubQueue = this.genApi.subscribeToQueue(scene.id, (item: QueueItem) => {
        if (item.status === 'completed' && item.result_url) {
          this.canvasService.addImageLayer('Generated', item.result_url);
        }
      });
    }
  }

  ngOnDestroy() {
    this.unsubQueue?.();
  }

  getGenerationResolution(): { w: number; h: number } {
    const frame = this.canvasService.frameRect();
    const ratio = frame.w / frame.h;
    const tier = this.resolutionTier();
    const maxPixels = tier === '1MP' ? 1048576 : 2097152;
    let w = Math.round(Math.sqrt(maxPixels * ratio));
    let h = Math.round(w / ratio);
    w = Math.round(w / 16) * 16;
    h = Math.round(h / 16) * 16;
    return { w, h };
  }

  toggleQwenSource(layerId: string) {
    if (this.qwenSourceLayers.has(layerId)) {
      this.qwenSourceLayers.delete(layerId);
    } else if (this.qwenSourceLayers.size < 3) {
      this.qwenSourceLayers.add(layerId);
    }
  }

  onRefImagesSelected(e: Event) {
    const input = e.target as HTMLInputElement;
    if (!input.files) return;
    const remaining = 2 - this.qwenRefImagesBase64.length;
    for (let i = 0; i < Math.min(input.files.length, remaining); i++) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        this.qwenRefImagesBase64.push(ev.target?.result as string);
      };
      reader.readAsDataURL(input.files[i]);
    }
    input.value = '';
  }

  removeRefImage(index: number) {
    this.qwenRefImagesBase64.splice(index, 1);
  }

  onMaskFileSelected(e: Event) {
    const input = e.target as HTMLInputElement;
    if (!input.files?.[0]) return;
    const reader = new FileReader();
    reader.onload = (ev) => { this.uploadedMaskBase64 = ev.target?.result as string; };
    reader.readAsDataURL(input.files[0]);
  }

  onControlImageSelected(e: Event) {
    const input = e.target as HTMLInputElement;
    if (!input.files?.[0]) return;
    const reader = new FileReader();
    reader.onload = (ev) => { this.uploadedControlBase64 = ev.target?.result as string; };
    reader.readAsDataURL(input.files[0]);
  }

  canGenerate(): boolean {
    if (this.activeMode() === 'qwen_edit') {
      return !!this.qwenInstruction.trim();
    }
    return !!this.prompt.trim();
  }

  async generate() {
    this.generationSuccess.set(false);

    // Qwen Edit — direct API call
    if (this.activeMode() === 'qwen_edit') {
      const scene = this.vellumService.selectedScene();
      if (!scene) return;

      const frameCapture = this.canvasService.exportFrameAsBase64();
      const result = await this.genApi.qwenEdit({
        instruction: this.qwenInstruction,
        source_images: [frameCapture],
        reference_images: this.qwenRefImagesBase64,
      });

      if (result) {
        await this.canvasService.addImageLayer('Qwen Edit Result', result);
        this.generationSuccess.set(true);
        setTimeout(() => this.generationSuccess.set(false), 5000);
      }
      return;
    }

    // Z-Image — merge preset + character/location LoRAs
    const mergedLoRAs = this.presetService.mergeLoRAs([]);
    const finalPrompt = this.presetService.buildFinalPrompt(this.prompt);

    const res = this.getGenerationResolution();
    const params: GenerationParams = {
      mode: this.activeMode(),
      prompt: finalPrompt,
      negative_prompt: '',
      model: 'turbo',
      width: res.w,
      height: res.h,
      steps: this.steps,
      seed: this.seed,
      scheduler: this.scheduler,
      guidance_scale: 0,
      loras: mergedLoRAs,
    };

    // Inpaint extras
    if (this.activeMode() === 'inpaint') {
      params.mask_blur = this.maskBlur;
      params.denoise_strength = this.denoiseStrength;
      params.input_image_base64 = this.canvasService.exportFrameAsBase64();

      if (this.maskSource === 'canvas') {
        params.mask_base64 = this.canvasService.exportMaskAsBase64();
      } else {
        params.mask_base64 = this.uploadedMaskBase64 ?? undefined;
      }
    }

    // ControlNet extras
    if (this.activeMode() === 'controlnet') {
      params.controlnet_type = this.selectedControlType();
      params.controlnet_scale = this.controlScale;

      if (this.controlImageSource === 'canvas') {
        params.control_image_base64 = this.canvasService.exportFrameAsBase64();
      } else {
        params.control_image_base64 = this.uploadedControlBase64 ?? undefined;
      }
    }

    const imageUrl = await this.genApi.generateDirect(params);
    if (imageUrl) {
      await this.canvasService.addImageLayer('Generated', imageUrl);
      this.generationSuccess.set(true);
      setTimeout(() => this.generationSuccess.set(false), 5000);
    }
  }
}
