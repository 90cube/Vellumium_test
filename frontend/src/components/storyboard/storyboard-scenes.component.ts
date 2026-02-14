
import { Component, inject, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { VellumiumService, Scene } from '../../services/vellumium.service';
import { CharacterService } from '../../services/character.service';
import { LocationService } from '../../services/location.service';
import { Character } from '../../models/character.model';
import { Location } from '../../models/location.model';

@Component({
  selector: 'app-storyboard-scenes',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="flex-1 relative flex overflow-hidden bg-[var(--cinema-bg)]">

      <!-- Timeline Axis (left) -->
      <div class="w-16 flex-shrink-0 flex flex-col items-center py-10 relative z-10 bg-[var(--cinema-surface)] border-r border-[var(--cinema-border)] overflow-y-auto no-scrollbar">
        <!-- Top line -->
        <div class="w-px bg-gradient-to-b from-transparent via-[var(--cinema-primary)]/20 to-[var(--cinema-primary)]/20 h-12"></div>

        @for (scene of scenes(); track scene.id; let idx = $index; let last = $last) {
          <!-- Node -->
          <div class="flex flex-col items-center cursor-pointer group relative" (click)="setActiveScene(idx)">
            @if (idx === activeSceneIndex()) {
              <!-- Active node: glowing dot -->
              <div class="w-4 h-4 rounded-full bg-[var(--cinema-primary)] shadow-[0_0_12px_var(--cinema-primary)] z-20 relative"></div>
              <div class="absolute w-8 h-8 rounded-full bg-[var(--cinema-primary)]/20 animate-pulse z-10"></div>
            } @else {
              <!-- Inactive node: hollow dot -->
              <div class="w-3 h-3 rounded-full border-2 border-[var(--cinema-text-muted)] bg-[var(--cinema-bg)] group-hover:border-[var(--cinema-primary)] group-hover:scale-110 transition-all z-20"></div>
            }
          </div>

          <!-- Connector -->
          @if (!last) {
            <div class="w-px bg-[var(--cinema-primary)]/20 my-1 transition-all duration-500" [class]="idx === activeSceneIndex() || idx + 1 === activeSceneIndex() ? 'h-32 bg-[var(--cinema-primary)]/40' : 'h-24'"></div>
          }
        }

        <!-- Empty state -->
        @if (scenes().length === 0) {
          <div class="flex-1 flex items-center justify-center py-4">
            <span class="text-[var(--cinema-text-muted)] text-[10px] rotate-90 whitespace-nowrap tracking-widest uppercase opacity-50">Timeline Start</span>
          </div>
        }

        <!-- Bottom fade -->
        <div class="w-px bg-gradient-to-b from-[var(--cinema-primary)]/20 to-transparent h-full flex-1 min-h-[40px]"></div>
      </div>

      <!-- Scene Cards (main area) -->
      <div class="flex-1 overflow-y-auto px-12 py-10 no-scrollbar scroll-smooth">
        <div class="flex flex-col items-center justify-start gap-8 min-h-full">

          @for (scene of scenes(); track scene.id; let idx = $index) {
            @if (idx === activeSceneIndex()) {
              <!-- ======== ACTIVE SCENE ======== -->
              <div class="w-full max-w-4xl relative z-10 transition-all duration-500 scale-100 group">
                <div class="glass-panel rounded-xl overflow-hidden cursor-pointer border border-[var(--cinema-primary)]/30 shadow-2xl shadow-[var(--cinema-primary)]/5" (click)="openScene(scene)">
                  <!-- Thumbnail -->
                  <div class="aspect-video w-full bg-black relative overflow-hidden">
                    @if (scene.thumbnail_url) {
                      <div class="absolute inset-0 bg-cover bg-center transition-transform duration-[20s] ease-linear hover:scale-105" [style.background-image]="'url(' + scene.thumbnail_url + ')'"></div>
                    } @else {
                      <div class="absolute inset-0 flex items-center justify-center bg-[var(--cinema-surface)]">
                        <span class="material-symbols-outlined text-6xl text-[var(--cinema-text-muted)]/20">movie</span>
                      </div>
                    }
                    <!-- Overlay -->
                    <div class="absolute inset-0 bg-gradient-to-t from-[var(--cinema-bg)] via-transparent to-transparent opacity-90"></div>

                    <!-- Active Shot Badge (top-left) -->
                    <div class="absolute top-5 left-5 flex items-center gap-2 bg-black/40 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10">
                      <span class="w-2 h-2 rounded-full bg-[#00ff9d] shadow-[0_0_8px_#00ff9d]"></span>
                      <span class="text-[10px] font-bold text-white uppercase tracking-wider">Active Scene</span>
                    </div>

                    <!-- Scene Title -->
                    <div class="absolute bottom-0 left-0 right-0 p-8 pt-20 bg-gradient-to-t from-[var(--cinema-bg)] to-transparent">
                      <h3 class="font-bold text-white text-3xl mb-2 flex items-center gap-3">
                        <span class="text-[var(--cinema-primary)] opacity-50 text-xl font-mono">0{{ idx + 1 }}</span>
                        {{ scene.title }}
                      </h3>
                      @if (scene.description) {
                        <p class="text-[var(--cinema-text-muted)] text-sm max-w-2xl leading-relaxed">{{ scene.description }}</p>
                      }
                    </div>

                    <!-- Hover Actions (top-right) -->
                    <div class="absolute top-5 right-5 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity transform translate-y-2 group-hover:translate-y-0 duration-300">
                      <button class="flex items-center justify-center w-10 h-10 rounded-full bg-white/10 hover:bg-white/20 backdrop-blur-md border border-white/10 text-white transition-colors shadow-lg" (click)="$event.stopPropagation(); openScene(scene)">
                        <span class="material-symbols-outlined text-[20px]">edit</span>
                      </button>
                    </div>
                  </div>
                </div>

                <!-- Scene Meta -->
                <div class="mt-4 flex items-center gap-4 px-2">
                  <div class="text-xs px-3 py-1.5 rounded-full bg-[var(--cinema-surface)] border border-[var(--cinema-border)] flex items-center gap-2 text-[var(--cinema-text-muted)]">
                    <span class="material-symbols-outlined text-[16px]">layers</span>
                    <span class="font-medium text-white">{{ scene.layers?.length || 0 }}</span> Layers
                  </div>
                </div>

                <!-- Binding Slots -->
                <div class="mt-3 flex gap-3 px-2">
                  <!-- Character slot -->
                  <div
                    class="flex-1 flex items-center gap-2 px-3 py-2 rounded-lg border transition-all min-h-[40px]"
                    [class]="dragOverSlot() === scene.id + '-character'
                      ? 'border-[var(--cinema-primary)] bg-[var(--cinema-primary)]/10'
                      : scene.character_id
                        ? 'border-[var(--cinema-border)] bg-[var(--cinema-surface)]'
                        : 'border-dashed border-[var(--cinema-border)] bg-transparent'"
                    (dragover)="onDragOver($event, scene.id, 'character')"
                    (dragleave)="onDragLeave()"
                    (drop)="onDrop($event, scene.id, 'character')"
                    (click)="scene.character_id ? unbindCharacter(scene) : null">
                    @if (getCharacter(scene.character_id); as char) {
                      @if (char.thumbnail_url) {
                        <img [src]="char.thumbnail_url" class="w-6 h-6 rounded object-cover border border-[var(--cinema-border)]">
                      } @else {
                        <span class="material-symbols-outlined text-[16px] text-[var(--cinema-primary)]">person</span>
                      }
                      <span class="text-[10px] text-white truncate">{{ char.name }}</span>
                      <span class="material-symbols-outlined text-[12px] text-[var(--cinema-text-muted)] ml-auto cursor-pointer hover:text-white">close</span>
                    } @else {
                      <span class="material-symbols-outlined text-[16px] text-[var(--cinema-text-dim)]">person</span>
                      <span class="text-[10px] text-[var(--cinema-text-dim)]">Drop character</span>
                    }
                  </div>

                  <!-- Location slot -->
                  <div
                    class="flex-1 flex items-center gap-2 px-3 py-2 rounded-lg border transition-all min-h-[40px]"
                    [class]="dragOverSlot() === scene.id + '-location'
                      ? 'border-[var(--cinema-primary)] bg-[var(--cinema-primary)]/10'
                      : scene.location_id
                        ? 'border-[var(--cinema-border)] bg-[var(--cinema-surface)]'
                        : 'border-dashed border-[var(--cinema-border)] bg-transparent'"
                    (dragover)="onDragOver($event, scene.id, 'location')"
                    (dragleave)="onDragLeave()"
                    (drop)="onDrop($event, scene.id, 'location')"
                    (click)="scene.location_id ? unbindLocation(scene) : null">
                    @if (getLocation(scene.location_id); as loc) {
                      @if (loc.thumbnail_url) {
                        <img [src]="loc.thumbnail_url" class="w-6 h-6 rounded object-cover border border-[var(--cinema-border)]">
                      } @else {
                        <span class="material-symbols-outlined text-[16px] text-[var(--cinema-primary)]">terrain</span>
                      }
                      <span class="text-[10px] text-white truncate">{{ loc.name }}</span>
                      <span class="material-symbols-outlined text-[12px] text-[var(--cinema-text-muted)] ml-auto cursor-pointer hover:text-white">close</span>
                    } @else {
                      <span class="material-symbols-outlined text-[16px] text-[var(--cinema-text-dim)]">terrain</span>
                      <span class="text-[10px] text-[var(--cinema-text-dim)]">Drop location</span>
                    }
                  </div>
                </div>
              </div>

            } @else if (getSceneDistance(idx) === 1) {
              <!-- ======== ADJACENT SCENE ======== -->
              <div
                class="w-full max-w-3xl relative group cursor-pointer transition-all duration-500 opacity-60 hover:opacity-90 scale-95 hover:scale-[0.97]"
                (click)="setActiveScene(idx)"
              >
                <div class="glass-panel rounded-xl overflow-hidden border border-[var(--cinema-border)]">
                  <div class="aspect-video w-full bg-[var(--cinema-surface)] relative">
                    @if (scene.thumbnail_url) {
                      <div class="absolute inset-0 bg-cover bg-center" [style.background-image]="'url(' + scene.thumbnail_url + ')'"></div>
                    }
                    <div class="absolute inset-0 bg-[var(--cinema-bg)]/60 group-hover:bg-[var(--cinema-bg)]/40 transition-colors"></div>
                    <div class="absolute bottom-6 left-6">
                      <h3 class="font-medium text-white/80 text-xl flex items-center gap-3">
                         <span class="text-[var(--cinema-text-muted)] text-sm font-mono">0{{ idx + 1 }}</span>
                         {{ scene.title }}
                      </h3>
                    </div>
                  </div>
                </div>

                <!-- Binding Slots (compact) -->
                <div class="mt-2 flex gap-2 px-1">
                  <div
                    class="flex-1 flex items-center gap-1.5 px-2 py-1.5 rounded-lg border transition-all"
                    [class]="dragOverSlot() === scene.id + '-character'
                      ? 'border-[var(--cinema-primary)] bg-[var(--cinema-primary)]/10'
                      : scene.character_id
                        ? 'border-[var(--cinema-border)] bg-[var(--cinema-surface)]'
                        : 'border-dashed border-[var(--cinema-border)]'"
                    (dragover)="onDragOver($event, scene.id, 'character')"
                    (dragleave)="onDragLeave()"
                    (drop)="onDrop($event, scene.id, 'character')"
                    (click)="scene.character_id ? unbindCharacter(scene) : null; $event.stopPropagation()">
                    <span class="material-symbols-outlined text-[14px]"
                      [class]="scene.character_id ? 'text-[var(--cinema-primary)]' : 'text-[var(--cinema-text-dim)]'">person</span>
                    <span class="text-[10px] truncate"
                      [class]="scene.character_id ? 'text-white/80' : 'text-[var(--cinema-text-dim)]'">{{ getCharacterName(scene.character_id) }}</span>
                  </div>
                  <div
                    class="flex-1 flex items-center gap-1.5 px-2 py-1.5 rounded-lg border transition-all"
                    [class]="dragOverSlot() === scene.id + '-location'
                      ? 'border-[var(--cinema-primary)] bg-[var(--cinema-primary)]/10'
                      : scene.location_id
                        ? 'border-[var(--cinema-border)] bg-[var(--cinema-surface)]'
                        : 'border-dashed border-[var(--cinema-border)]'"
                    (dragover)="onDragOver($event, scene.id, 'location')"
                    (dragleave)="onDragLeave()"
                    (drop)="onDrop($event, scene.id, 'location')"
                    (click)="scene.location_id ? unbindLocation(scene) : null; $event.stopPropagation()">
                    <span class="material-symbols-outlined text-[14px]"
                      [class]="scene.location_id ? 'text-[var(--cinema-primary)]' : 'text-[var(--cinema-text-dim)]'">terrain</span>
                    <span class="text-[10px] truncate"
                      [class]="scene.location_id ? 'text-white/80' : 'text-[var(--cinema-text-dim)]'">{{ getLocationName(scene.location_id) }}</span>
                  </div>
                </div>
              </div>

            } @else {
              <!-- ======== DISTANT SCENE ======== -->
              <div
                class="w-full max-w-2xl relative opacity-30 hover:opacity-50 scale-90 transition-all duration-500 cursor-pointer"
                (click)="setActiveScene(idx)"
              >
                <div class="aspect-video w-full bg-[var(--cinema-surface)] rounded-xl overflow-hidden relative border border-[var(--cinema-border)]">
                  @if (scene.thumbnail_url) {
                    <div class="absolute inset-0 bg-cover bg-center grayscale" [style.background-image]="'url(' + scene.thumbnail_url + ')'"></div>
                    <div class="absolute inset-0 bg-[var(--cinema-bg)]/80"></div>
                  } @else {
                    <div class="absolute inset-0 flex items-center justify-center">
                      <span class="text-[var(--cinema-text-dim)] font-mono text-lg">0{{ idx + 1 }}</span>
                    </div>
                  }
                </div>

                <!-- Binding indicators (minimal) -->
                <div class="mt-1.5 flex gap-2 justify-center">
                  @if (scene.character_id) {
                    <span class="material-symbols-outlined text-[12px] text-[var(--cinema-primary)]/60">person</span>
                  }
                  @if (scene.location_id) {
                    <span class="material-symbols-outlined text-[12px] text-[var(--cinema-primary)]/60">terrain</span>
                  }
                </div>
              </div>
            }
          }

          <!-- Empty State -->
          @if (scenes().length === 0) {
            <div class="text-center py-20">
               <div class="w-16 h-16 rounded-2xl bg-[var(--cinema-surface)] border border-[var(--cinema-border)] flex items-center justify-center mx-auto mb-4 shadow-lg">
                  <span class="material-symbols-outlined text-3xl text-[var(--cinema-text-muted)]">movie</span>
               </div>
              <h3 class="text-lg font-medium text-white mb-2">No Scenes Yet</h3>
              <p class="text-[var(--cinema-text-muted)] text-sm">Add your first scene to begin creating your story.</p>
            </div>
          }

          <!-- Add Scene Card -->
          <div class="w-full max-w-3xl mt-4" (click)="showAddScene.set(true)">
            <div class="aspect-video w-full rounded-xl border border-dashed border-[var(--cinema-border)] flex flex-col items-center justify-center cursor-pointer hover:border-[var(--cinema-primary)]/50 hover:bg-[var(--cinema-primary)]/5 transition-all group">
              <div class="w-12 h-12 rounded-full bg-[var(--cinema-surface)] flex items-center justify-center mb-3 group-hover:scale-110 transition-transform shadow-lg">
                  <span class="material-symbols-outlined text-2xl text-[var(--cinema-text-muted)] group-hover:text-[var(--cinema-primary)]">add</span>
              </div>
              <span class="text-[var(--cinema-text-muted)] group-hover:text-white text-sm font-medium transition-colors">Add New Scene</span>
            </div>
          </div>

        </div>
      </div>

      <!-- Add Scene Modal -->
      @if (showAddScene()) {
        <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm animate-fade-in" (click)="showAddScene.set(false)">
          <div class="relative w-[420px] glass-panel rounded-xl shadow-2xl p-6 animate-scale-in border border-[var(--cinema-border)]" (click)="$event.stopPropagation()">
            
            <div class="flex items-center justify-between mb-6">
                <h3 class="text-lg font-bold text-white">New Scene</h3>
                <button (click)="showAddScene.set(false)" class="text-[var(--cinema-text-muted)] hover:text-white transition-colors">
                  <span class="material-symbols-outlined">close</span>
                </button>
            </div>

            <div class="flex flex-col gap-4">
              <div>
                <label class="text-xs font-medium text-[var(--cinema-text-muted)] mb-1.5 block">Scene Title</label>
                <input
                  type="text"
                  [(ngModel)]="newSceneTitle"
                  placeholder="e.g. The Alley, The Chase..."
                  class="w-full px-4 py-2.5 bg-[var(--cinema-bg)] border border-[var(--cinema-border)] rounded-lg text-white text-sm focus:border-[var(--cinema-primary)] focus:outline-none transition-colors placeholder-[var(--cinema-text-dim)]"
                />
              </div>

              <div>
                <label class="text-xs font-medium text-[var(--cinema-text-muted)] mb-1.5 block">Description</label>
                <textarea
                  [(ngModel)]="newSceneDescription"
                  placeholder="Scene description..."
                  rows="3"
                  class="w-full px-4 py-2.5 bg-[var(--cinema-bg)] border border-[var(--cinema-border)] rounded-lg text-white text-sm focus:border-[var(--cinema-primary)] focus:outline-none transition-colors placeholder-[var(--cinema-text-dim)] resize-none"
                ></textarea>
              </div>

              @if (vellumService.error()) {
                <div class="text-red-400 text-xs text-center bg-red-500/10 p-2 rounded">{{ vellumService.error() }}</div>
              }

              <button
                (click)="onAddScene()"
                [disabled]="!newSceneTitle.trim() || isCreating()"
                class="w-full py-2.5 bg-[var(--cinema-primary)] hover:bg-[#6015c7] text-white font-bold rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-[var(--cinema-primary)]/20 mt-2">
                {{ isCreating() ? 'ADDING...' : 'Add Scene' }}
              </button>
            </div>
          </div>
        </div>
      }
    </div>
  `,
  styles: [`
    .animate-scale-in {
      animation: scaleIn 0.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .animate-fade-in {
       animation: fadeIn 0.2s ease-out forwards;
    }
    @keyframes scaleIn {
      from { transform: scale(0.95); opacity: 0; }
      to { transform: scale(1); opacity: 1; }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .no-scrollbar::-webkit-scrollbar { display: none; }
    .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
  `]
})
export class StoryboardScenesComponent {
  vellumService = inject(VellumiumService);
  private characterService = inject(CharacterService);
  private locationService = inject(LocationService);

  showAddScene = signal(false);
  isCreating = signal(false);
  activeSceneIndex = signal(0);
  dragOverSlot = signal<string | null>(null);
  newSceneTitle = '';
  newSceneDescription = '';

  scenes = computed(() => this.vellumService.selectedProject()?.scenes ?? []);

  getSceneDistance(idx: number): number {
    return Math.abs(idx - this.activeSceneIndex());
  }

  // --- Binding lookups ---
  getCharacter(id: string | null): Character | null {
    if (!id) return null;
    return this.characterService.characters().find(c => c.id === id) ?? null;
  }

  getLocation(id: string | null): Location | null {
    if (!id) return null;
    return this.locationService.locations().find(l => l.id === id) ?? null;
  }

  getCharacterName(id: string | null): string {
    const char = this.getCharacter(id);
    return char ? char.name : 'Character';
  }

  getLocationName(id: string | null): string {
    const loc = this.getLocation(id);
    return loc ? loc.name : 'Location';
  }

  // --- Drag & Drop ---
  onDragOver(event: DragEvent, sceneId: string, slotType: 'character' | 'location') {
    event.preventDefault();
    event.dataTransfer!.dropEffect = 'link';
    this.dragOverSlot.set(sceneId + '-' + slotType);
  }

  onDragLeave() {
    this.dragOverSlot.set(null);
  }

  onDrop(event: DragEvent, sceneId: string, slotType: 'character' | 'location') {
    event.preventDefault();
    this.dragOverSlot.set(null);

    const raw = event.dataTransfer?.getData('application/json');
    if (!raw) return;

    try {
      const payload = JSON.parse(raw) as { type: string; id: string };
      if (payload.type === 'character' && slotType === 'character') {
        this.vellumService.bindCharacterToScene(sceneId, payload.id);
      } else if (payload.type === 'location' && slotType === 'location') {
        this.vellumService.bindLocationToScene(sceneId, payload.id);
      }
    } catch { /* ignore bad data */ }
  }

  unbindCharacter(scene: Scene) {
    this.vellumService.bindCharacterToScene(scene.id, null);
  }

  unbindLocation(scene: Scene) {
    this.vellumService.bindLocationToScene(scene.id, null);
  }

  setActiveScene(idx: number) {
    this.activeSceneIndex.set(idx);
  }

  openScene(scene: Scene) {
    this.vellumService.selectScene(scene);
  }

  async onAddScene() {
    const project = this.vellumService.selectedProject();
    if (!project || !this.newSceneTitle.trim()) return;

    this.isCreating.set(true);
    const scene = await this.vellumService.createScene(
      project.id,
      this.newSceneTitle.trim(),
      this.newSceneDescription.trim()
    );

    this.isCreating.set(false);
    if (scene) {
      this.showAddScene.set(false);
      this.newSceneTitle = '';
      this.newSceneDescription = '';
      const newScenes = this.vellumService.selectedProject()?.scenes ?? [];
      this.activeSceneIndex.set(newScenes.length - 1);
    }
  }
}
