
import { Component, inject, effect, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { CharacterService } from '../../services/character.service';
import { VellumiumService } from '../../services/vellumium.service';
import { GenerationApiService } from '../../services/generation-api.service';
import { Character, CHARACTER_SLIDER_DEFS } from '../../models/character.model';
import { STYLE_PRESETS, StylePreset } from '../../models/style-presets';

@Component({
  selector: 'app-character-workspace',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="h-full flex bg-[var(--cinema-bg)]">

      <!-- Left Column: Editor + Thumbnails -->
      <div class="w-80 flex-shrink-0 flex flex-col border-r border-[var(--cinema-border)] bg-[var(--cinema-surface)]">

        <!-- Editor (scrollable) -->
        @if (characterService.selectedCharacter(); as char) {
          <div class="flex-1 overflow-y-auto p-5 space-y-4">

            <!-- Name -->
            <div class="space-y-1">
              <label class="text-[10px] font-display font-semibold text-[var(--cinema-text-muted)] uppercase tracking-wider">Name</label>
              <input
                type="text"
                [ngModel]="char.name"
                (ngModelChange)="editName = $event"
                class="w-full bg-[var(--cinema-bg)] text-[var(--cinema-text)] text-xs border border-[var(--cinema-border)] rounded-lg px-3 py-2 outline-none focus:border-[var(--cinema-primary)]/30">
            </div>

            <!-- Attribute Sliders -->
            <div class="space-y-2">
              <label class="text-[10px] font-display font-semibold text-[var(--cinema-text-muted)] uppercase tracking-wider">Attributes</label>
              @for (slider of sliderDefs; track slider.id) {
                <div class="space-y-0.5">
                  <div class="flex items-center justify-between">
                    <span class="text-[10px] text-[var(--cinema-text)]">{{ slider.label }}</span>
                    <span class="text-[10px] font-mono w-10 text-right"
                      [class]="getSliderValue(char, slider.id) === 0
                        ? 'text-[var(--cinema-text-dim)]'
                        : 'text-[var(--cinema-primary)]'">{{ getSliderValue(char, slider.id).toFixed(2) }}</span>
                  </div>
                  <div class="flex items-center gap-1.5">
                    <span class="text-[9px] text-[var(--cinema-text-dim)] w-10 text-right shrink-0">{{ slider.negLabel }}</span>
                    <input type="range" min="-200" max="200" step="1"
                      [value]="getSliderValue(char, slider.id) * 100"
                      (input)="onSliderChange(slider.id, +$any($event.target).value / 100)"
                      (dblclick)="onSliderChange(slider.id, 0)"
                      class="flex-1 h-1 accent-[var(--cinema-primary)] cursor-pointer">
                    <span class="text-[9px] text-[var(--cinema-text-dim)] w-10 shrink-0">{{ slider.posLabel }}</span>
                  </div>
                </div>
              }
            </div>

            <!-- Prompt -->
            <div class="space-y-1.5">
              <label class="text-[10px] font-display font-semibold text-[var(--cinema-text-muted)] uppercase tracking-wider">Description</label>
              <textarea
                [ngModel]="char.prompt"
                (ngModelChange)="editPrompt = $event"
                class="w-full h-20 bg-[var(--cinema-bg)] text-[var(--cinema-text)] text-xs border border-[var(--cinema-border)] rounded-lg px-3 py-2 focus:ring-1 focus:ring-[var(--cinema-primary)]/30 focus:border-[var(--cinema-primary)]/50 focus:outline-none resize-none"
                placeholder="Character description..."></textarea>
            </div>

            <!-- Generate + Save -->
            @if (genError()) {
              <div class="text-[var(--cinema-error)] text-[10px]">{{ genError() }}</div>
            }
            <button
              (click)="generatePortrait(char)"
              [disabled]="isGenerating()"
              class="w-full py-2.5 text-[11px] font-display font-semibold rounded-lg transition-all flex items-center justify-center gap-1.5
                bg-gradient-to-r from-[var(--cinema-primary)] to-[var(--cinema-accent)] text-white
                hover:shadow-lg hover:shadow-[var(--cinema-primary)]/20
                disabled:opacity-40 disabled:cursor-not-allowed">
              <span class="material-symbols-outlined text-[16px]">auto_awesome</span>
              {{ isGenerating() ? 'Generating...' : 'Generate Portrait' }}
            </button>

            <div class="flex gap-2">
              <button
                (click)="saveCharacter(char)"
                class="flex-1 py-1.5 text-[11px] font-display font-semibold rounded-lg bg-[var(--cinema-primary)]/15 text-[var(--cinema-primary)] border border-[var(--cinema-primary)]/30 hover:bg-[var(--cinema-primary)]/25 transition-colors flex items-center justify-center gap-1">
                <span class="material-symbols-outlined text-[14px]">save</span>
                Save
              </button>
              <button
                (click)="deleteCharacter(char)"
                class="px-3 py-1.5 text-[11px] font-display rounded-lg text-[var(--cinema-text-muted)] border border-[var(--cinema-border)] hover:text-[var(--cinema-error)] hover:border-[var(--cinema-error)]/30 transition-colors">
                <span class="material-symbols-outlined text-[14px]">delete</span>
              </button>
            </div>
          </div>
        } @else {
          <!-- No character selected: prompt -->
          <div class="flex-1 flex items-center justify-center">
            <div class="text-center space-y-3">
              <span class="material-symbols-outlined text-4xl text-[var(--cinema-text-dim)]">person</span>
              <p class="text-sm text-[var(--cinema-text-muted)]">Select a character<br>or create a new one</p>
            </div>
          </div>
        }

        <!-- Thumbnail Strip (bottom) -->
        <div class="border-t border-[var(--cinema-border)] p-3">
          <div class="flex gap-2 overflow-x-auto no-scrollbar items-center">
            @for (char of characterService.characters(); track char.id) {
              <div
                class="shrink-0 w-14 h-14 rounded-lg border-2 cursor-pointer transition-all overflow-hidden"
                [class]="characterService.selectedCharacter()?.id === char.id
                  ? 'border-[var(--cinema-primary)] shadow-[0_0_8px_var(--cinema-primary)]/30'
                  : 'border-[var(--cinema-border)] hover:border-[var(--cinema-primary)]/40'"
                (click)="selectCharacter(char)"
                draggable="true"
                (dragstart)="onDragStart($event, char)">
                @if (char.thumbnail_url) {
                  <img [src]="char.thumbnail_url" class="w-full h-full object-cover">
                } @else {
                  <div class="w-full h-full bg-[var(--cinema-bg)] flex items-center justify-center">
                    <span class="material-symbols-outlined text-[20px] text-[var(--cinema-text-dim)]">person</span>
                  </div>
                }
              </div>
            }
            <!-- Add button -->
            <button
              (click)="addCharacter()"
              class="shrink-0 w-14 h-14 rounded-lg border-2 border-dashed border-[var(--cinema-border)] flex items-center justify-center hover:border-[var(--cinema-primary)]/40 hover:bg-[var(--cinema-primary)]/5 transition-all cursor-pointer">
              <span class="material-symbols-outlined text-[20px] text-[var(--cinema-text-muted)]">add</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Right Column: Full-size Preview -->
      <div class="flex-1 flex items-center justify-center p-8 relative overflow-hidden">
        @if (characterService.selectedCharacter(); as char) {
          @if (char.thumbnail_url) {
            <!-- Full-size image -->
            <div class="relative max-h-full flex flex-col items-center">
              <img
                [src]="char.thumbnail_url"
                class="max-h-[calc(100vh-200px)] w-auto rounded-xl border border-[var(--cinema-border)] shadow-2xl object-contain"
                [alt]="char.name">

              <!-- Character info overlay (bottom) -->
              <div class="mt-4 text-center max-w-md">
                <h2 class="text-xl font-display font-bold text-white mb-1">{{ char.name }}</h2>
                @if (char.prompt) {
                  <p class="text-sm text-[var(--cinema-text-muted)] leading-relaxed">{{ char.prompt }}</p>
                }
                @if (getProjectStyleLabel()) {
                  <span class="inline-block mt-2 px-2.5 py-0.5 text-[10px] rounded-full bg-[var(--cinema-primary)]/15 text-[var(--cinema-primary)] border border-[var(--cinema-primary)]/30">{{ getProjectStyleLabel() }}</span>
                }
              </div>
            </div>
          } @else {
            <!-- No image yet -->
            <div class="text-center space-y-4">
              <div class="w-24 h-24 rounded-2xl bg-[var(--cinema-surface)] border border-[var(--cinema-border)] flex items-center justify-center mx-auto">
                <span class="material-symbols-outlined text-5xl text-[var(--cinema-text-dim)]">person</span>
              </div>
              <div>
                <h3 class="text-lg font-display font-medium text-white mb-1">{{ char.name }}</h3>
                <p class="text-sm text-[var(--cinema-text-muted)]">Generate a portrait to visualize this character</p>
              </div>
            </div>
          }
        } @else {
          <!-- Nothing selected -->
          <div class="text-center space-y-4 opacity-40">
            <span class="material-symbols-outlined text-6xl text-[var(--cinema-text-dim)]">group</span>
            <p class="text-[var(--cinema-text-muted)]">Select or create a character to get started</p>
          </div>
        }

        <!-- Generating overlay -->
        @if (isGenerating()) {
          <div class="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-10">
            <div class="text-center space-y-3">
              <div class="w-12 h-12 border-2 border-[var(--cinema-primary)] border-t-transparent rounded-full animate-spin mx-auto"></div>
              <p class="text-sm text-[var(--cinema-text-muted)]">Generating portrait...</p>
            </div>
          </div>
        }
      </div>
    </div>
  `,
  styles: [`
    .no-scrollbar::-webkit-scrollbar { display: none; }
    .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
  `]
})
export class CharacterWorkspaceComponent {
  characterService = inject(CharacterService);
  private vellumiumService = inject(VellumiumService);
  private genApi = inject(GenerationApiService);

  sliderDefs = CHARACTER_SLIDER_DEFS;

  editName: string | null = null;
  editPrompt: string | null = null;
  editSliders: Record<string, number> = {};

  isGenerating = signal(false);
  genError = signal<string | null>(null);

  private projectLoadEffect = effect(() => {
    const project = this.vellumiumService.selectedProject();
    if (project) {
      this.characterService.loadCharacters(project.id);
    }
  });

  selectCharacter(char: Character) {
    this.characterService.selectedCharacter.set(char);
    this.editName = null;
    this.editPrompt = null;
    this.editSliders = { ...char.sliders };
  }

  getSliderValue(char: Character, sliderId: string): number {
    if (sliderId in this.editSliders) {
      return this.editSliders[sliderId];
    }
    return char.sliders[sliderId] ?? 0;
  }

  onSliderChange(sliderId: string, value: number) {
    this.editSliders[sliderId] = value;
  }

  getProjectStyleLabel(): string | null {
    const project = this.vellumiumService.selectedProject();
    if (!project?.style_preset_id) return null;
    return STYLE_PRESETS.find(s => s.id === project.style_preset_id)?.label ?? null;
  }

  private getProjectStyle(): StylePreset | undefined {
    const project = this.vellumiumService.selectedProject();
    if (!project?.style_preset_id) return undefined;
    return STYLE_PRESETS.find(s => s.id === project.style_preset_id);
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
    this.editName = null;
    this.editPrompt = null;
  }

  async deleteCharacter(char: Character) {
    await this.characterService.deleteCharacter(char.id);
  }

  async generatePortrait(char: Character) {
    this.isGenerating.set(true);
    this.genError.set(null);

    const promptParts: string[] = [
      'full body portrait, front view, facing the camera, standing, full length shot',
    ];

    const style = this.getProjectStyle();
    if (style?.promptKeywords) {
      promptParts.push(style.promptKeywords);
    }

    const userPrompt = (this.editPrompt ?? char.prompt).trim();
    if (userPrompt) {
      promptParts.push(userPrompt);
    }

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
      await this.characterService.updateCharacter(char.id, { thumbnail_url: imageUrl });
    } else {
      this.genError.set(this.genApi.lastError() ?? 'Generation failed');
    }
  }

  onDragStart(event: DragEvent, char: Character) {
    event.dataTransfer?.setData('application/json', JSON.stringify({ type: 'character', id: char.id }));
    event.dataTransfer!.effectAllowed = 'link';
  }
}
