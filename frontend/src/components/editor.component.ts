
import { Component, inject, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { VellumiumService, Scene, ProjectStage } from '../services/vellumium.service';
import { StoryboardScenesComponent } from './storyboard/storyboard-scenes.component';
import { DirectorsCutScenesComponent } from './directors-cut/directorscut-scenes.component';
import { FilmEditorComponent } from './film/film-editor.component';
import { LocationPanelComponent } from './storyboard/location-panel.component';
import { CharacterWorkspaceComponent } from './storyboard/character-workspace.component';

type SidebarTab = 'scenes' | 'characters' | 'locations' | 'assets' | 'settings';

@Component({
  selector: 'app-editor',
  standalone: true,
  imports: [CommonModule, FormsModule, StoryboardScenesComponent, DirectorsCutScenesComponent, FilmEditorComponent, LocationPanelComponent, CharacterWorkspaceComponent],
  template: `
    <div class="h-screen flex bg-[var(--cinema-bg)] text-[var(--cinema-text)] font-sans overflow-hidden">

      <!-- Sidebar -->
      <aside class="w-64 flex-shrink-0 flex flex-col bg-[var(--cinema-surface)] border-r border-[var(--cinema-border)] h-full z-30">
        <!-- Project Header -->
        <div class="p-6 border-b border-[var(--cinema-border)]">
          <h1 class="font-bold text-lg leading-tight truncate text-white">{{ vellumService.selectedProject()?.title }}</h1>
          <p class="text-[var(--cinema-text-muted)] text-xs mt-1">Editor Workspace</p>
        </div>

        <!-- Nav Links -->
        <nav class="flex flex-col gap-1 p-3">
          <a class="flex items-center gap-3 px-4 py-2.5 rounded-lg text-[var(--cinema-text-muted)] hover:text-white hover:bg-white/5 transition-colors cursor-pointer"
             (click)="vellumService.backToDashboard()">
            <span class="material-symbols-outlined text-[20px]">arrow_back</span>
            <span class="text-sm font-medium">Back to Projects</span>
          </a>

          <div class="h-px bg-[var(--cinema-border)] my-2 mx-4"></div>

          @for (tab of sidebarTabs; track tab.id) {
            <a class="flex items-center gap-3 px-4 py-2.5 rounded-lg transition-colors cursor-pointer"
               [class]="activeSidebarTab() === tab.id
                 ? 'bg-[var(--cinema-primary)]/10 text-[var(--cinema-primary)]'
                 : 'text-[var(--cinema-text-muted)] hover:text-white hover:bg-white/5'"
               (click)="activeSidebarTab.set(tab.id)">
              <span class="material-symbols-outlined text-[20px]" [class.fill-1]="activeSidebarTab() === tab.id">{{ tab.icon }}</span>
              <span class="text-sm font-medium">{{ tab.label }}</span>
              @if (tab.id === 'scenes') {
                <span class="ml-auto bg-[var(--cinema-primary)]/20 text-[var(--cinema-primary)] text-[10px] font-bold px-2 py-0.5 rounded-full">{{ scenes().length }}</span>
              }
            </a>
          }
        </nav>

        <!-- Sidebar Panel Content (only for views that need sidebar editing) -->
        <div class="flex-1 overflow-hidden border-t border-[var(--cinema-border)]">
          @switch (activeSidebarTab()) {
            @case ('locations') { <app-location-panel class="block h-full" /> }
          }
        </div>
      </aside>

      <!-- Main Content Area -->
      <main class="flex-1 flex flex-col relative min-w-0 bg-[var(--cinema-bg)]">

        @if (activeSidebarTab() === 'characters') {
          <!-- Character Workspace (full main area) -->
          <app-character-workspace class="flex-1" />
        } @else {
          <!-- Top Bar (scenes/stage view) -->
          <header class="h-14 flex-shrink-0 flex items-center justify-between px-6 border-b border-[var(--cinema-border)] bg-[var(--cinema-surface)]/80 backdrop-blur-md z-20">

            <!-- Stage Tabs -->
            <div class="flex items-center gap-1">
              @for (tab of stageTabs; track tab.stage) {
                <button
                  (click)="vellumService.switchStage(tab.stage)"
                  class="flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-all duration-200"
                  [class]="vellumService.currentStage() === tab.stage
                    ? 'bg-[var(--cinema-primary)] text-white shadow-sm shadow-[var(--cinema-primary)]/20'
                    : 'text-[var(--cinema-text-muted)] hover:text-white hover:bg-white/5'"
                >
                  <span class="material-symbols-outlined text-[18px]">{{ tab.icon }}</span>
                  <span>{{ tab.label }}</span>
                </button>
              }
            </div>

            <div class="flex items-center gap-2">
              <button class="p-2 rounded-lg text-[var(--cinema-text-muted)] hover:text-white hover:bg-white/5 transition-colors">
                <span class="material-symbols-outlined text-[20px]">help</span>
              </button>
              <button (click)="showAddScene.set(true)" class="px-3 py-1.5 bg-[var(--cinema-surface)] border border-[var(--cinema-border)] hover:bg-[var(--cinema-elevated)] text-xs font-medium rounded text-white transition-colors flex items-center gap-2">
                 <span class="material-symbols-outlined text-[16px]">add</span>
                 New Scene
              </button>
            </div>
          </header>

          <!-- Stage Content -->
          <div class="flex-1 overflow-hidden relative">
              @switch (vellumService.currentStage()) {
              @case ('STORYBOARD') {
                  <app-storyboard-scenes />
              }
              @case ('DIRECTORS_CUT') {
                  <app-directorscut-scenes />
              }
              @case ('FILM') {
                  <app-film-editor />
              }
              }
          </div>
        }
      </main>

      <!-- Add Scene Modal -->
      @if (showAddScene()) {
        <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm animate-fade-in" (click)="showAddScene.set(false)">
          <div class="relative w-[420px] glass-panel rounded-xl shadow-2xl p-6 border border-[var(--cinema-border)] animate-scale-in" (click)="$event.stopPropagation()">
            
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
  `]
})
export class EditorComponent {
  vellumService = inject(VellumiumService);

  activeSidebarTab = signal<SidebarTab>('scenes');
  showAddScene = signal(false);
  isCreating = signal(false);
  newSceneTitle = '';
  newSceneDescription = '';

  scenes = computed(() => this.vellumService.selectedProject()?.scenes ?? []);

  readonly sidebarTabs: { id: SidebarTab; label: string; icon: string }[] = [
    { id: 'scenes', label: 'Scenes', icon: 'movie' },
    { id: 'characters', label: 'Characters', icon: 'person' },
    { id: 'locations', label: 'Locations', icon: 'terrain' },
    { id: 'assets', label: 'Assets', icon: 'image' },
    { id: 'settings', label: 'Settings', icon: 'settings' },
  ];

  readonly stageTabs: { stage: ProjectStage; label: string; icon: string }[] = [
    { stage: 'STORYBOARD', label: 'Storyboard', icon: 'draw' },
    { stage: 'DIRECTORS_CUT', label: "Director's Cut", icon: 'movie_filter' },
    { stage: 'FILM', label: 'Film', icon: 'theaters' },
  ];

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
    }
  }
}
