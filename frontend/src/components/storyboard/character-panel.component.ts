
import { Component, inject, effect, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { CharacterService } from '../../services/character.service';
import { VellumiumService } from '../../services/vellumium.service';
import { GenerationApiService } from '../../services/generation-api.service';
import { Character, CHARACTER_SLIDER_DEFS } from '../../models/character.model';
import { STYLE_PRESETS, StylePreset } from '../../models/style-presets';

@Component({
  selector: 'app-character-panel',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="flex flex-col h-full">

      <!-- Header -->
      <div class="flex items-center justify-between px-4 py-3 border-b border-[--cinema-border]">
        <h3 class="text-xs font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Characters</h3>
        <button
          (click)="addCharacter()"
          class="p-1 rounded hover:bg-[--cinema-primary]/10 text-[--cinema-text-muted] hover:text-[--cinema-primary] transition-colors"
          title="Add Character">
          <span class="material-symbols-outlined text-[16px]">person_add</span>
        </button>
      </div>

      <!-- Character List (scrollable grid) -->
      <div class="overflow-y-auto p-3"
        [class]="characterService.selectedCharacter() ? 'max-h-[160px]' : 'flex-1'">
        @if (characterService.characters().length === 0) {
          <div class="text-xs text-[--cinema-text-dim] text-center py-6 font-display">No characters yet</div>
        } @else {
          <div class="grid grid-cols-3 gap-2">
            @for (char of characterService.characters(); track char.id) {
              <div
                draggable="true"
                (dragstart)="onDragStart($event, char)"
                class="flex flex-col items-center gap-1 p-2 rounded-lg border cursor-pointer transition-all"
                [class]="characterService.selectedCharacter()?.id === char.id
                  ? 'bg-[--cinema-primary]/10 border-[--cinema-primary]/30'
                  : 'bg-[--cinema-surface] border-[--cinema-border] hover:border-[--cinema-primary]/20'"
                (click)="selectCharacter(char)">
                <!-- Thumbnail or placeholder -->
                @if (char.thumbnail_url) {
                  <img [src]="char.thumbnail_url" class="w-12 h-12 rounded-lg object-cover border border-[--cinema-border]">
                } @else {
                  <div class="w-12 h-12 rounded-lg bg-[--cinema-bg] border border-[--cinema-border] flex items-center justify-center">
                    <span class="material-symbols-outlined text-[20px] text-[--cinema-text-dim]">person</span>
                  </div>
                }
                <span class="text-[10px] text-[--cinema-text] truncate w-full text-center">{{ char.name }}</span>
              </div>
            }
          </div>
        }
      </div>

      <!-- Editor (when character selected) -->
      @if (characterService.selectedCharacter(); as char) {
        <div class="flex-1 overflow-y-auto border-t border-[--cinema-border] p-4 space-y-4">

          <!-- Name -->
          <div class="space-y-1">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Name</label>
            <input
              type="text"
              [ngModel]="char.name"
              (ngModelChange)="editName = $event"
              class="w-full bg-[--cinema-bg] text-[--cinema-text] text-xs border border-[--cinema-border] rounded-lg px-3 py-1.5 outline-none focus:border-[--cinema-primary]/30">
          </div>

          <!-- Style Presets (mutually exclusive pills) -->
          <div class="space-y-1.5">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Style</label>
            <div class="flex flex-wrap gap-1">
              @for (style of stylePresets; track style.id) {
                <button
                  class="px-2 py-1 text-[10px] rounded-full border transition-colors"
                  [class]="char.style_preset_id === style.id
                    ? 'bg-[--cinema-primary]/15 text-[--cinema-primary] border-[--cinema-primary]/30'
                    : 'text-[--cinema-text-muted] border-[--cinema-border] hover:border-[--cinema-primary]/30'"
                  (click)="toggleStylePreset(char, style.id)">{{ style.label }}</button>
              }
            </div>
          </div>

          <!-- Character Sliders -->
          <div class="space-y-2">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Attributes</label>
            @for (slider of sliderDefs; track slider.id) {
              <div class="space-y-0.5">
                <div class="flex items-center justify-between">
                  <span class="text-[10px] text-[--cinema-text]">{{ slider.label }}</span>
                  <span class="text-[10px] font-mono w-10 text-right"
                    [class]="getSliderValue(char, slider.id) === 0
                      ? 'text-[--cinema-text-dim]'
                      : 'text-[--cinema-primary]'">{{ getSliderValue(char, slider.id).toFixed(2) }}</span>
                </div>
                <div class="flex items-center gap-1.5">
                  <span class="text-[9px] text-[--cinema-text-dim] w-10 text-right shrink-0">{{ slider.negLabel }}</span>
                  <input type="range" min="-200" max="200" step="1"
                    [value]="getSliderValue(char, slider.id) * 100"
                    (input)="onSliderChange(char, slider.id, +$any($event.target).value / 100)"
                    (dblclick)="onSliderChange(char, slider.id, 0)"
                    class="flex-1 h-1 accent-[--cinema-primary] cursor-pointer">
                  <span class="text-[9px] text-[--cinema-text-dim] w-10 shrink-0">{{ slider.posLabel }}</span>
                </div>
              </div>
            }
          </div>

          <!-- Prompt -->
          <div class="space-y-1.5">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Prompt</label>
            <textarea
              [ngModel]="char.prompt"
              (ngModelChange)="editPrompt = $event"
              class="w-full h-16 bg-[--cinema-bg] text-[--cinema-text] text-xs border border-[--cinema-border] rounded-lg px-3 py-2 focus:ring-1 focus:ring-[--cinema-primary]/30 focus:border-[--cinema-primary]/50 focus:outline-none resize-none"
              placeholder="Character description..."></textarea>
          </div>

          <!-- Generate Portrait -->
          <div class="space-y-1.5">
            @if (genError()) {
              <div class="text-[--cinema-error] text-[10px] px-1">{{ genError() }}</div>
            }
            @if (genSuccess()) {
              <div class="text-green-400 text-[10px] px-1">Portrait generated.</div>
            }
            <button
              (click)="generatePortrait(char)"
              [disabled]="isGenerating()"
              class="w-full py-2 text-[11px] font-display font-semibold rounded-lg transition-all flex items-center justify-center gap-1.5
                bg-gradient-to-r from-[--cinema-primary] to-[--cinema-accent] text-white
                hover:shadow-lg hover:shadow-[--cinema-primary]/20
                disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none">
              <span class="material-symbols-outlined text-[16px]">auto_awesome</span>
              {{ isGenerating() ? 'Generating...' : 'Generate Portrait' }}
            </button>
          </div>

          <!-- Actions -->
          <div class="flex gap-2">
            <button
              (click)="saveCharacter(char)"
              class="flex-1 py-1.5 text-[11px] font-display font-semibold rounded-lg bg-[--cinema-primary]/15 text-[--cinema-primary] border border-[--cinema-primary]/30 hover:bg-[--cinema-primary]/25 transition-colors flex items-center justify-center gap-1">
              <span class="material-symbols-outlined text-[14px]">save</span>
              Save
            </button>
            <button
              (click)="deleteCharacter(char)"
              class="px-3 py-1.5 text-[11px] font-display rounded-lg bg-[--cinema-panel] text-[--cinema-text-muted] border border-[--cinema-border] hover:text-[--cinema-error] hover:border-[--cinema-error]/30 transition-colors flex items-center justify-center gap-1">
              <span class="material-symbols-outlined text-[14px]">delete</span>
              Delete
            </button>
          </div>
        </div>
      }
    </div>
  `
})
export class CharacterPanelComponent {
  characterService = inject(CharacterService);
  vellumiumService = inject(VellumiumService);
  private genApi = inject(GenerationApiService);

  stylePresets = STYLE_PRESETS;
  sliderDefs = CHARACTER_SLIDER_DEFS;

  // Local edit buffers (null means use value from selectedCharacter)
  editName: string | null = null;
  editPrompt: string | null = null;
  editSliders: Record<string, number> = {};

  // Generation state
  isGenerating = signal(false);
  genError = signal<string | null>(null);
  genSuccess = signal(false);

  private projectLoadEffect = effect(() => {
    const project = this.vellumiumService.selectedProject();
    if (project) {
      this.characterService.loadCharacters(project.id);
    }
  });

  selectCharacter(char: Character) {
    this.characterService.selectedCharacter.set(char);
    // Reset local edit buffers
    this.editName = null;
    this.editPrompt = null;
    this.editSliders = { ...char.sliders };
  }

  getSliderValue(char: Character, sliderId: string): number {
    // Prefer local edit buffer if slider was changed
    if (sliderId in this.editSliders) {
      return this.editSliders[sliderId];
    }
    return char.sliders[sliderId] ?? 0;
  }

  onSliderChange(char: Character, sliderId: string, value: number) {
    this.editSliders[sliderId] = value;
  }

  toggleStylePreset(char: Character, styleId: string) {
    const newValue = char.style_preset_id === styleId ? null : styleId;
    this.characterService.updateCharacter(char.id, { style_preset_id: newValue });
  }

  async addCharacter() {
    const project = this.vellumiumService.selectedProject();
    if (!project) return;
    const created = await this.characterService.createCharacter(project.id, 'New Character');
    if (created) {
      this.selectCharacter(created);
    }
  }

  async saveCharacter(char: Character) {
    const updates: Partial<Character> = {};

    if (this.editName !== null && this.editName !== char.name) {
      updates.name = this.editName;
    }
    if (this.editPrompt !== null && this.editPrompt !== char.prompt) {
      updates.prompt = this.editPrompt;
    }
    if (Object.keys(this.editSliders).length > 0) {
      updates.sliders = { ...char.sliders, ...this.editSliders };
    }

    if (Object.keys(updates).length > 0) {
      await this.characterService.updateCharacter(char.id, updates);
    }

    // Reset edit buffers
    this.editName = null;
    this.editPrompt = null;
  }

  onDragStart(event: DragEvent, char: Character) {
    event.dataTransfer?.setData('application/json', JSON.stringify({ type: 'character', id: char.id }));
    event.dataTransfer!.effectAllowed = 'link';
  }

  async deleteCharacter(char: Character) {
    await this.characterService.deleteCharacter(char.id);
  }

  async generatePortrait(char: Character) {
    this.isGenerating.set(true);
    this.genError.set(null);
    this.genSuccess.set(false);

    // Build prompt: full body front view first, then style keywords, then user prompt
    const promptParts: string[] = [
      'full body portrait, front view, facing the camera, standing, full length shot',
    ];

    // Style preset keywords
    const style: StylePreset | undefined = char.style_preset_id
      ? STYLE_PRESETS.find(s => s.id === char.style_preset_id)
      : undefined;
    if (style?.promptKeywords) {
      promptParts.push(style.promptKeywords);
    }

    // User's character description
    const userPrompt = (this.editPrompt ?? char.prompt).trim();
    if (userPrompt) {
      promptParts.push(userPrompt);
    }

    // Build LoRAs: style LoRA + character slider LoRAs
    const loras: { filename: string; scale: number }[] = [];
    if (style) {
      for (const lora of style.loras) {
        loras.push({ filename: lora.filename, scale: lora.scale });
      }
    }
    const sliders = { ...char.sliders, ...this.editSliders };
    for (const def of CHARACTER_SLIDER_DEFS) {
      const val = sliders[def.id] ?? 0;
      if (val !== 0) {
        loras.push({ filename: def.loraFilename, scale: val });
      }
    }

    const imageUrl = await this.genApi.generateDirect({
      mode: 't2i',
      prompt: promptParts.join(', '),
      negative_prompt: '',
      model: 'turbo',
      width: 768,
      height: 1280,
      steps: 8,
      seed: -1,
      scheduler: 'Euler',
      guidance_scale: 0,
      loras,
    });

    this.isGenerating.set(false);

    if (imageUrl) {
      // Save as character thumbnail
      await this.characterService.updateCharacter(char.id, { thumbnail_url: imageUrl });
      this.genSuccess.set(true);
      setTimeout(() => this.genSuccess.set(false), 5000);
    } else {
      this.genError.set(this.genApi.lastError() ?? 'Generation failed');
    }
  }
}
