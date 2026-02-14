
import { Component, inject, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VellumiumService, Scene } from '../../services/vellumium.service';

@Component({
  selector: 'app-film-editor',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="h-screen flex flex-col bg-[--cinema-bg] overflow-hidden">

      <!-- ═══ HEADER ═══ -->
      <header class="h-14 flex items-center justify-between bg-[--cinema-bg] border-b border-[--cinema-gold]/10 px-5 shrink-0 z-50">
        <!-- Left: Back -->
        <div class="flex items-center gap-3">
          <button
            class="btn-cinema flex items-center justify-center size-9"
            (click)="vellumService.backToEditor()">
            <span class="material-symbols-outlined text-lg">arrow_back</span>
          </button>
        </div>

        <!-- Center: Title + Badge -->
        <div class="flex items-center gap-3">
          <h2 class="font-cinema text-[--cinema-gold] text-lg tracking-wide">
            {{ vellumService.selectedProject()?.title || 'Untitled' }}
          </h2>
          <span class="bg-[--cinema-gold]/10 text-[--cinema-gold] text-xs rounded px-2 py-0.5 font-display uppercase tracking-wider">
            Film Editor
          </span>
        </div>

        <!-- Right: Export + Settings -->
        <div class="flex items-center gap-2">
          <button class="btn-velvet flex items-center gap-1.5 px-3 py-1.5 text-sm opacity-50 cursor-not-allowed" disabled>
            <span class="material-symbols-outlined text-sm">movie</span>
            Export
          </button>
          <button class="btn-cinema flex items-center justify-center size-9">
            <span class="material-symbols-outlined text-lg">settings</span>
          </button>
        </div>
      </header>

      <!-- ═══ MAIN CONTENT ═══ -->
      <div class="flex-1 flex overflow-hidden">

        <!-- ── Preview Monitor (left ~60%) ── -->
        <section class="flex-[3] flex flex-col items-center justify-center min-w-0 relative"
                 style="background: linear-gradient(180deg, color-mix(in srgb, var(--cinema-gold) 3%, transparent) 0%, transparent 40%), var(--cinema-bg);">

          @if (videoScenes().length === 0) {
            <!-- Empty state -->
            <div class="brass-frame p-12 relative">
              <div class="art-deco-corner"></div>
              <div class="flex flex-col items-center gap-4 text-center px-8">
                <span class="material-symbols-outlined text-5xl text-[--cinema-gold]/20">theaters</span>
                <p class="font-cinema text-[--cinema-gold]/30 text-base leading-relaxed">
                  No processed footage.<br/>Visit Director's Cut first.
                </p>
              </div>
            </div>
          } @else {
            <div class="w-full max-w-3xl px-8">
              <!-- Preview Frame -->
              <div class="brass-frame relative aspect-video overflow-hidden">
                @if (getSelectedClip(); as clip) {
                  @if (clip.video_thumbnail_url) {
                    <img [src]="clip.video_thumbnail_url" class="w-full h-full object-cover" alt="Preview" />
                  } @else if (clip.confirmed_image_url) {
                    <img [src]="clip.confirmed_image_url" class="w-full h-full object-cover" alt="Preview" />
                  } @else {
                    <div class="w-full h-full bg-[--cinema-bg] flex items-center justify-center">
                      <span class="material-symbols-outlined text-6xl text-[--cinema-gold]/10">theaters</span>
                    </div>
                  }
                } @else if (videoScenes()[0]; as first) {
                  @if (first.video_thumbnail_url) {
                    <img [src]="first.video_thumbnail_url" class="w-full h-full object-cover" alt="Preview" />
                  } @else {
                    <div class="w-full h-full bg-[--cinema-bg] flex items-center justify-center">
                      <span class="material-symbols-outlined text-6xl text-[--cinema-gold]/10">theaters</span>
                    </div>
                  }
                }

                <!-- Play overlay -->
                <div class="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity duration-300 cursor-pointer">
                  <span class="material-symbols-outlined text-[--cinema-gold]/40 hover:text-[--cinema-gold]/80 transition-colors duration-200"
                        style="font-size: 64px; filter: drop-shadow(0 0 12px color-mix(in srgb, var(--cinema-gold) 30%, transparent));">
                    play_circle
                  </span>
                </div>
              </div>

              <!-- Timecode Display -->
              <div class="flex items-center justify-center gap-3 mt-4">
                <span class="font-mono text-[--cinema-gold] text-sm tracking-widest">
                  {{ formatTimecode(playheadPosition()) }}
                </span>
                <span class="text-[--cinema-gold]/30 text-sm">/</span>
                <span class="font-mono text-[--cinema-gold]/50 text-sm tracking-widest">
                  {{ formatTimecode(100) }}
                </span>
              </div>
              <!-- Format pills -->
              <div class="flex items-center justify-center gap-2 mt-2">
                <span class="text-[--cinema-gold]/20 text-[10px] font-mono border border-[--cinema-gold]/10 rounded px-1.5 py-0.5">REC.709</span>
                <span class="text-[--cinema-gold]/20 text-[10px] font-mono border border-[--cinema-gold]/10 rounded px-1.5 py-0.5">1080p</span>
              </div>
            </div>
          }
        </section>

        <!-- ── Properties Panel (right ~40%) ── -->
        <aside class="w-96 flex flex-col bg-[--cinema-panel] border-l border-[--cinema-gold]/10 min-w-0">
          <div class="px-4 pt-4 pb-3">
            <h3 class="font-display uppercase text-xs text-[--cinema-gold]/60 tracking-wider">Editing Tracks</h3>
          </div>

          <div class="flex-1 overflow-y-auto px-4 pb-4 space-y-3">

            <!-- Video Track Card -->
            <div class="cinema-panel rounded-lg p-3">
              <div class="flex items-center gap-2 mb-3">
                <span class="material-symbols-outlined text-base text-[--cinema-gold]">film_reel</span>
                <span class="font-display font-semibold text-[--cinema-gold] text-sm">Video</span>
                <span class="text-[10px] text-[--cinema-text-muted] ml-1">{{ videoScenes().length }} clips</span>
                <div class="flex-1"></div>
                <button class="size-7 flex items-center justify-center rounded text-[--cinema-gold]/40 hover:text-[--cinema-gold] transition-colors">
                  <span class="material-symbols-outlined text-sm">visibility</span>
                </button>
                <button class="size-7 flex items-center justify-center rounded text-[--cinema-gold]/40 hover:text-[--cinema-gold] transition-colors">
                  <span class="material-symbols-outlined text-sm">lock_open</span>
                </button>
              </div>
              <!-- Clip thumbnails -->
              <div class="flex gap-1.5 overflow-x-auto pb-1">
                @for (scene of videoScenes(); track scene.id) {
                  <div
                    class="shrink-0 w-20 h-[45px] rounded overflow-hidden relative cursor-pointer transition-all"
                    [class]="selectedClipId() === scene.id
                      ? 'brass-frame cinema-glow border-[--cinema-gold]'
                      : 'border border-[--cinema-border] hover:border-[--cinema-gold]/40'"
                    (click)="selectClip(scene.id, $event)">
                    @if (scene.video_thumbnail_url) {
                      <img [src]="scene.video_thumbnail_url" class="w-full h-full object-cover" alt="" />
                    } @else if (scene.confirmed_image_url) {
                      <img [src]="scene.confirmed_image_url" class="w-full h-full object-cover" alt="" />
                    } @else {
                      <div class="w-full h-full bg-[--cinema-bg] flex items-center justify-center">
                        <span class="material-symbols-outlined text-xs text-[--cinema-gold]/20">theaters</span>
                      </div>
                    }
                    <!-- Scene title overlay -->
                    <div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent px-1 py-0.5">
                      <span class="text-[10px] text-[--cinema-text] truncate block">{{ scene.title }}</span>
                    </div>
                    @if (scene.clip_trimmed) {
                      <span class="absolute top-0.5 right-0.5 material-symbols-outlined text-[10px] text-[--cinema-gold]">content_cut</span>
                    }
                  </div>
                }
                @if (videoScenes().length === 0) {
                  <span class="text-xs text-[--cinema-text-dim]">No clips</span>
                }
              </div>
            </div>

            <!-- Audio Track Card -->
            <div class="cinema-panel rounded-lg p-3">
              <div class="flex items-center gap-2 mb-3">
                <span class="material-symbols-outlined text-base text-[--cinema-gold]">music_note</span>
                <span class="font-display font-semibold text-[--cinema-gold] text-sm">Audio</span>
                <div class="flex-1"></div>
                <button class="size-7 flex items-center justify-center rounded text-[--cinema-gold]/40 hover:text-[--cinema-gold] transition-colors">
                  <span class="material-symbols-outlined text-sm">visibility</span>
                </button>
                <button class="size-7 flex items-center justify-center rounded text-[--cinema-gold]/40 hover:text-[--cinema-gold] transition-colors">
                  <span class="material-symbols-outlined text-sm">lock_open</span>
                </button>
              </div>
              <button class="btn-cinema w-full py-2 text-xs border-dashed">
                <span class="material-symbols-outlined text-sm mr-1 align-middle">add</span>
                Add Audio Track
              </button>
            </div>

            <!-- Subtitle Track Card -->
            <div class="cinema-panel rounded-lg p-3">
              <div class="flex items-center gap-2 mb-3">
                <span class="material-symbols-outlined text-base text-[--cinema-gold]">closed_caption</span>
                <span class="font-display font-semibold text-[--cinema-gold] text-sm">Subtitles</span>
                <div class="flex-1"></div>
                <button class="size-7 flex items-center justify-center rounded text-[--cinema-gold]/40 hover:text-[--cinema-gold] transition-colors">
                  <span class="material-symbols-outlined text-sm">visibility</span>
                </button>
                <button class="size-7 flex items-center justify-center rounded text-[--cinema-gold]/40 hover:text-[--cinema-gold] transition-colors">
                  <span class="material-symbols-outlined text-sm">lock_open</span>
                </button>
              </div>
              <button class="btn-cinema w-full py-2 text-xs border-dashed">
                <span class="material-symbols-outlined text-sm mr-1 align-middle">add</span>
                Add Subtitle Track
              </button>
            </div>

          </div>
        </aside>
      </div>

      <!-- ═══ BOTTOM TIMELINE ═══ -->
      <div class="h-52 bg-[--cinema-bg] border-t border-[--cinema-gold]/10 flex flex-col shrink-0">

        <!-- Toolbar Row -->
        <div class="h-10 flex items-center px-4 gap-2 shrink-0">
          <!-- Scissors toggle -->
          <button
            class="btn-cinema flex items-center justify-center size-8 transition-all"
            [class]="scissorsActive()
              ? 'bg-[--cinema-gold]/20 text-[--cinema-gold] cinema-glow border-[--cinema-gold]/40'
              : ''"
            (click)="toggleScissors()">
            <span class="material-symbols-outlined text-sm">content_cut</span>
          </button>
          @if (scissorsActive()) {
            <span class="text-[--cinema-gold]/40 text-xs">Select a clip to trim</span>
          }

          <!-- Zoom -->
          <button class="btn-cinema flex items-center justify-center size-7">
            <span class="material-symbols-outlined text-sm">zoom_in</span>
          </button>
          <button class="btn-cinema flex items-center justify-center size-7">
            <span class="material-symbols-outlined text-sm">zoom_out</span>
          </button>

          <div class="flex-1"></div>

          <!-- Playback controls -->
          <button class="btn-cinema flex items-center justify-center size-7">
            <span class="material-symbols-outlined text-sm">skip_previous</span>
          </button>
          <button class="btn-cinema flex items-center justify-center size-7">
            <span class="material-symbols-outlined text-sm">fast_rewind</span>
          </button>
          <button class="btn-velvet flex items-center justify-center size-8 rounded-full">
            <span class="material-symbols-outlined text-lg">play_arrow</span>
          </button>
          <button class="btn-cinema flex items-center justify-center size-7">
            <span class="material-symbols-outlined text-sm">fast_forward</span>
          </button>
          <button class="btn-cinema flex items-center justify-center size-7">
            <span class="material-symbols-outlined text-sm">skip_next</span>
          </button>

          <div class="flex-1"></div>

          <!-- Timecode -->
          <span class="font-mono text-[--cinema-gold]/60 text-xs tracking-wider">
            {{ formatTimecode(playheadPosition()) }}
          </span>
        </div>

        <!-- Timeline Ruler -->
        <div class="relative h-6 bg-[--cinema-panel] shrink-0 cursor-pointer"
             (click)="onRulerClick($event)">
          <div class="absolute inset-0 flex items-end pl-20 pr-4">
            @for (mark of timeMarks; track mark.label) {
              <div class="absolute bottom-0 flex flex-col items-center" [style.left]="'calc(' + mark.pos + '% * 0.8 + 80px)'">
                <span class="text-[10px] text-[--cinema-gold]/30 mb-0.5 font-mono">{{ mark.label }}</span>
                <div class="w-px h-2 bg-[--cinema-gold]/20"></div>
              </div>
            }
          </div>
          <!-- Playhead marker on ruler -->
          <div class="absolute top-0 bottom-0 w-0.5 bg-[--cinema-gold] z-10"
               [style.left]="'calc(' + playheadPosition() + '% * 0.8 + 80px)'">
            <div class="absolute -top-0 -left-1 w-0 h-0 border-l-[5px] border-r-[5px] border-t-[6px] border-l-transparent border-r-transparent border-t-[--cinema-gold]"></div>
          </div>
        </div>

        <!-- Track Lanes -->
        <div class="flex-1 overflow-hidden relative">
          <!-- Playhead line -->
          <div class="absolute top-0 bottom-0 w-0.5 bg-[--cinema-gold] z-20 pointer-events-none"
               [style.left]="'calc(' + playheadPosition() + '% * 0.8 + 80px)'"></div>

          <!-- Track labels column -->
          <div class="absolute left-0 top-0 bottom-0 w-20 bg-[--cinema-bg] border-r border-[--cinema-border]/50 z-10 flex flex-col">
            <div class="h-12 flex items-center gap-1.5 px-2 border-b border-[--cinema-border]/50">
              <span class="material-symbols-outlined text-xs text-[--cinema-gold]/30">film_reel</span>
              <span class="text-[10px] text-[--cinema-gold]/30 font-display">Video</span>
            </div>
            <div class="h-12 flex items-center gap-1.5 px-2 border-b border-[--cinema-border]/50">
              <span class="material-symbols-outlined text-xs text-[--cinema-gold]/30">music_note</span>
              <span class="text-[10px] text-[--cinema-gold]/30 font-display">Audio</span>
            </div>
            <div class="h-12 flex items-center gap-1.5 px-2">
              <span class="material-symbols-outlined text-xs text-[--cinema-gold]/30">closed_caption</span>
              <span class="text-[10px] text-[--cinema-gold]/30 font-display">Subs</span>
            </div>
          </div>

          <!-- Lane contents -->
          <div class="ml-20 h-full flex flex-col">
            <!-- Video lane -->
            <div class="h-12 flex items-center gap-1 px-2 border-b border-[--cinema-border]/50">
              @for (scene of videoScenes(); track scene.id) {
                <div
                  class="shrink-0 h-[80%] rounded-md flex items-center gap-1.5 px-2 cursor-pointer transition-all relative overflow-hidden"
                  [style.min-width]="getClipWidth(scene)"
                  [class]="selectedClipId() === scene.id
                    ? 'border border-[--cinema-gold]/60 cinema-glow'
                    : 'border border-[--cinema-gold]/20 hover:brightness-125'"
                  style="background: linear-gradient(135deg, #1A1508 0%, #2A1F0A 100%);"
                  (click)="selectClip(scene.id, $event)">
                  @if (scene.video_thumbnail_url) {
                    <img [src]="scene.video_thumbnail_url" class="h-full w-8 object-cover rounded-sm shrink-0 opacity-70" alt="" />
                  }
                  <span class="text-[10px] text-[--cinema-text] truncate">{{ scene.title }}</span>
                  @if (scene.clip_trimmed) {
                    <!-- Diagonal hash at trimmed edges -->
                    @if (scene.clip_start_ms > 0) {
                      <div class="absolute left-0 top-0 bottom-0 w-2"
                           style="background: repeating-linear-gradient(45deg, transparent, transparent 2px, color-mix(in srgb, var(--cinema-gold) 15%, transparent) 2px, color-mix(in srgb, var(--cinema-gold) 15%, transparent) 4px);"></div>
                    }
                    @if (scene.clip_end_ms !== null) {
                      <div class="absolute right-0 top-0 bottom-0 w-2"
                           style="background: repeating-linear-gradient(45deg, transparent, transparent 2px, color-mix(in srgb, var(--cinema-gold) 15%, transparent) 2px, color-mix(in srgb, var(--cinema-gold) 15%, transparent) 4px);"></div>
                    }
                  }
                </div>
                <!-- Gap between clips -->
                @if (!$last) {
                  <div class="shrink-0 w-1 h-[60%] bg-[--cinema-surface] rounded"></div>
                }
              }
              @if (videoScenes().length === 0) {
                <span class="text-xs text-[--cinema-text-dim]">No clips</span>
              }
            </div>

            <!-- Audio lane -->
            <div class="h-12 flex items-center px-2 border-b border-[--cinema-border]/50">
              <div class="w-full h-[80%] rounded-md border border-dashed border-[--cinema-border]/30 flex items-center justify-center">
                <span class="text-[10px] text-[--cinema-text-dim]">Drop audio here</span>
              </div>
            </div>

            <!-- Subtitle lane -->
            <div class="h-12 flex items-center px-2">
              <div class="w-full h-[80%] rounded-md border border-dashed border-[--cinema-border]/30 flex items-center justify-center">
                <span class="text-[10px] text-[--cinema-text-dim]">Drop subtitles here</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `
})
export class FilmEditorComponent {
  readonly vellumService = inject(VellumiumService);

  readonly activeTrack = signal<'video' | 'audio' | 'subtitle'>('video');
  readonly playheadPosition = signal(0);
  readonly scissorsActive = signal(false);
  readonly selectedClipId = signal<string | null>(null);

  readonly videoScenes = computed(() => {
    const project = this.vellumService.selectedProject();
    if (!project) return [];
    return project.scenes
      .filter(s => s.generated_video_url)
      .sort((a, b) => a.sort_order - b.sort_order);
  });

  readonly timeMarks = [
    { label: '0:00', pos: 0 },
    { label: '0:05', pos: 16.6 },
    { label: '0:10', pos: 33.3 },
    { label: '0:15', pos: 50 },
    { label: '0:20', pos: 66.6 },
    { label: '0:25', pos: 83.3 },
    { label: '0:30', pos: 100 },
  ];

  getSelectedClip(): Scene | null {
    const id = this.selectedClipId();
    if (!id) return null;
    return this.videoScenes().find(s => s.id === id) ?? null;
  }

  getClipWidth(scene: Scene): string {
    const duration = (scene.clip_end_ms ?? 5000) - scene.clip_start_ms;
    const minWidth = Math.max(80, duration / 50);
    return minWidth + 'px';
  }

  toggleScissors() {
    this.scissorsActive.set(!this.scissorsActive());
  }

  selectClip(id: string, event: Event) {
    event.stopPropagation();
    this.selectedClipId.set(this.selectedClipId() === id ? null : id);
  }

  onRulerClick(event: MouseEvent) {
    const target = event.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();
    const offsetX = event.clientX - rect.left - 80;
    const usableWidth = rect.width - 80 - 16;
    if (usableWidth > 0 && offsetX >= 0) {
      const percent = Math.min(100, Math.max(0, (offsetX / usableWidth) * 100));
      this.playheadPosition.set(percent);
    }
  }

  formatTimecode(percent: number): string {
    const totalSeconds = (percent / 100) * 30;
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = Math.floor(totalSeconds % 60);
    const frames = Math.floor((totalSeconds % 1) * 24);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}:${frames.toString().padStart(2, '0')}`;
  }
}
