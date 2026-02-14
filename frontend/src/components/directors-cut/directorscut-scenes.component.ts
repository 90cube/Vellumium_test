
import { Component, inject, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VellumiumService, Scene } from '../../services/vellumium.service';

@Component({
  selector: 'app-directorscut-scenes',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (scenes().length === 0) {
      <div class="flex flex-col items-center justify-center py-24 text-center">
        <div class="w-20 h-20 rounded-2xl bg-[var(--cinema-surface)] border border-[var(--cinema-border)] flex items-center justify-center mb-6 shadow-xl">
           <span class="material-symbols-outlined text-4xl text-[var(--cinema-text-muted)]">videocam_off</span>
        </div>
        <h3 class="text-lg font-medium mb-2 text-white">No scenes yet</h3>
        <p class="text-[var(--cinema-text-muted)] text-sm">Create scenes in the Storyboard first.</p>
      </div>
    } @else {
      <div class="flex flex-col gap-4 max-w-4xl mx-auto p-6">
        @for (scene of scenes(); track scene.id; let idx = $index) {
          <div
            class="flex glass-panel rounded-xl overflow-hidden cursor-pointer transition-all duration-300 hover:border-[var(--cinema-primary)]/50 hover:shadow-lg hover:shadow-[var(--cinema-primary)]/10 hover:-translate-y-1 group"
            (click)="onSelectScene(scene)"
          >
            <!-- Thumbnail -->
            <div class="w-48 flex-shrink-0 relative aspect-video overflow-hidden bg-black">
              @if (sceneThumb(scene)) {
                <div class="absolute inset-0">
                  <div class="absolute inset-0 bg-cover bg-center transition-transform duration-700 group-hover:scale-105" [style.background-image]="'url(' + sceneThumb(scene) + ')'"></div>
                </div>
                @if (scene.generated_video_url) {
                  <div class="absolute inset-0 flex items-center justify-center z-10 bg-black/30 backdrop-blur-[1px]">
                    <span class="material-symbols-outlined text-white text-[48px] drop-shadow-lg opacity-90 group-hover:opacity-100 group-hover:scale-110 transition-all">play_circle</span>
                  </div>
                }
              } @else {
                <div class="absolute inset-0 bg-[var(--cinema-surface)] flex items-center justify-center">
                  <span class="font-bold text-[var(--cinema-text-dim)] text-2xl">{{ idx + 1 }}</span>
                </div>
              }
            </div>
            <!-- Info -->
            <div class="flex-1 flex flex-col justify-center px-6 py-4 min-w-0">
              <h4 class="font-bold text-lg text-white truncate mb-1 group-hover:text-[var(--cinema-primary)] transition-colors">{{ scene.title || 'Untitled Scene' }}</h4>
              @if (scene.description) {
                <p class="text-[var(--cinema-text-muted)] text-sm truncate mb-3">{{ scene.description }}</p>
              }
              <!-- Status Badge -->
              @if (scene.generated_video_url) {
                <span class="inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-green-400 bg-green-900/20 border border-green-500/20 px-2.5 py-1 rounded w-fit">
                  <span class="material-symbols-outlined text-[12px]">verified</span>
                  Processed
                </span>
              } @else if (scene.confirmed_image_url) {
                <span class="inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-amber-400 bg-amber-900/20 border border-amber-500/20 px-2.5 py-1 rounded w-fit">
                  <span class="material-symbols-outlined text-[12px]">hourglass_top</span>
                  Ready to Film
                </span>
              } @else {
                <span class="inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-[var(--cinema-text-dim)] bg-[var(--cinema-surface)] border border-[var(--cinema-border)] px-2.5 py-1 rounded w-fit">
                  <span class="material-symbols-outlined text-[12px]">pending</span>
                  Draft
                </span>
              }
            </div>
            <!-- Arrow -->
            <div class="flex items-center pr-6">
              <span class="material-symbols-outlined text-[var(--cinema-text-muted)] text-[20px] group-hover:text-white transition-colors group-hover:translate-x-1">chevron_right</span>
            </div>
          </div>
        }
      </div>
    }
  `
})
export class DirectorsCutScenesComponent {
  private vellumService = inject(VellumiumService);

  scenes = computed(() => this.vellumService.selectedProject()?.scenes ?? []);

  sceneThumb(scene: Scene): string | null {
    return scene.video_thumbnail_url ?? scene.confirmed_image_url ?? scene.thumbnail_url;
  }

  onSelectScene(scene: Scene) {
    this.vellumService.selectSceneForVideogen(scene);
  }
}
