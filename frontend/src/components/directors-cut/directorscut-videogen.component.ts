
import { Component, inject, computed, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { VellumiumService, Scene } from '../../services/vellumium.service';

type I2vMethod = 'i2v' | 'first_last';

@Component({
  selector: 'app-directorscut-videogen',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="h-screen flex flex-col bg-[--cinema-bg] overflow-hidden">

      <!-- Header -->
      <header class="h-16 flex-shrink-0 flex items-center justify-between px-6 border-b border-[--cinema-gold]/10 bg-[--cinema-bg] z-20">
        <div class="flex items-center gap-4">
          <button
            (click)="onBack()"
            class="btn-cinema flex items-center gap-2 px-3 py-2 rounded-lg"
          >
            <span class="material-symbols-outlined text-[20px]">arrow_back</span>
            <span class="text-sm font-medium">Back</span>
          </button>
          <div class="w-px h-6 bg-[--cinema-gold]/10"></div>
          <div class="flex flex-col">
            <h2 class="font-cinema text-[--cinema-gold] text-lg">{{ scene()?.title || 'Untitled Scene' }}</h2>
            <p class="text-[--cinema-text-muted] text-xs">The Projection Booth</p>
          </div>
        </div>
        <!-- Status Badge -->
        <div>
          @if (scene()?.generated_video_url) {
            <span class="inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-green-400 bg-green-900/20 border border-green-700/30 px-3 py-1.5 rounded-lg">
              <span class="material-symbols-outlined text-[14px]">verified</span>
              Processed
            </span>
          } @else if (scene()?.confirmed_image_url) {
            <span class="inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-amber-400 bg-amber-900/20 border border-amber-700/30 px-3 py-1.5 rounded-lg">
              <span class="material-symbols-outlined text-[14px]">hourglass_top</span>
              Ready for Processing
            </span>
          } @else {
            <span class="inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-[--cinema-text-dim] bg-[--cinema-surface] border border-[--cinema-border] px-3 py-1.5 rounded-lg">
              <span class="material-symbols-outlined text-[14px]">pending</span>
              Awaiting Storyboard
            </span>
          }
        </div>
      </header>

      <!-- Main Content -->
      <div class="flex-1 flex gap-6 p-6 overflow-hidden">

        <!-- Left Panel: Source Frame -->
        <div class="flex-1 flex flex-col min-w-0">
          <h3 class="font-display uppercase tracking-wider text-xs text-[--cinema-gold]/60 mb-3">Source Frame</h3>
          <div class="flex-1 flex items-center justify-center brass-frame rounded-xl overflow-hidden bg-[--cinema-panel]">
            @if (scene()?.confirmed_image_url) {
              <img
                [src]="scene()!.confirmed_image_url!"
                alt="Source frame"
                class="max-w-full max-h-full object-contain"
              />
            } @else {
              <div class="flex flex-col items-center gap-4 text-center art-deco-corner p-12">
                <span class="material-symbols-outlined text-[--cinema-gold]/20 text-5xl">image</span>
                <p class="text-[--cinema-gold]/30 font-cinema">No confirmed image from Storyboard</p>
              </div>
            }
          </div>
          <p class="text-[--cinema-text-muted] text-xs mt-2">From Storyboard Scene #{{ (scene()?.sort_order ?? 0) + 1 }}</p>
        </div>

        <!-- Right Panel: Processing Controls -->
        <div class="w-96 flex-shrink-0 cinema-panel rounded-xl p-6 flex flex-col overflow-y-auto no-scrollbar">

          <!-- Generation Method -->
          <div class="mb-6">
            <h4 class="font-display uppercase text-xs text-[--cinema-gold]/60 tracking-wider mb-3">Generation Method</h4>
            <div class="flex flex-col gap-3">
              <div
                (click)="method.set('i2v')"
                class="cinema-panel rounded-lg p-4 cursor-pointer transition-all duration-200"
                [class.border-[--cinema-gold]/40]="method() === 'i2v'"
                [class.bg-[--cinema-gold]/5]="method() === 'i2v'"
                [class.border-[--cinema-border]]="method() !== 'i2v'"
                [class.hover:border-[--cinema-gold]/20]="method() !== 'i2v'"
              >
                <div class="flex items-center gap-3">
                  <span class="material-symbols-outlined text-[20px]" [class.text-[--cinema-gold]]="method() === 'i2v'" [class.text-[--cinema-text-muted]]="method() !== 'i2v'">movie</span>
                  <div>
                    <p class="text-sm font-medium" [class.text-[--cinema-gold]]="method() === 'i2v'" [class.text-[--cinema-text-muted]]="method() !== 'i2v'">Image to Video</p>
                    <p class="text-xs text-[--cinema-text-muted]/60">Transform still frame into motion</p>
                  </div>
                </div>
              </div>
              <div
                (click)="method.set('first_last')"
                class="cinema-panel rounded-lg p-4 cursor-pointer transition-all duration-200"
                [class.border-[--cinema-gold]/40]="method() === 'first_last'"
                [class.bg-[--cinema-gold]/5]="method() === 'first_last'"
                [class.border-[--cinema-border]]="method() !== 'first_last'"
                [class.hover:border-[--cinema-gold]/20]="method() !== 'first_last'"
              >
                <div class="flex items-center gap-3">
                  <span class="material-symbols-outlined text-[20px]" [class.text-[--cinema-gold]]="method() === 'first_last'" [class.text-[--cinema-text-muted]]="method() !== 'first_last'">compare</span>
                  <div>
                    <p class="text-sm font-medium" [class.text-[--cinema-gold]]="method() === 'first_last'" [class.text-[--cinema-text-muted]]="method() !== 'first_last'">First & Last Frame</p>
                    <p class="text-xs text-[--cinema-text-muted]/60">Define start and end frames</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="gold-line mb-6"></div>

          <!-- First & Last Frame inputs -->
          @if (method() === 'first_last') {
            <div class="mb-6">
              <div class="flex gap-3">
                <!-- First Frame -->
                <div class="flex-1 flex flex-col items-center gap-1.5">
                  <div class="w-full aspect-video rounded-lg brass-frame overflow-hidden flex items-center justify-center bg-[--cinema-panel]">
                    @if (scene()?.confirmed_image_url) {
                      <img [src]="scene()!.confirmed_image_url!" class="w-full h-full object-cover" alt="First frame" />
                    } @else {
                      <span class="material-symbols-outlined text-[--cinema-gold]/20 text-2xl">first_page</span>
                    }
                  </div>
                  <span class="text-[10px] text-[--cinema-text-muted] font-mono uppercase">First Frame</span>
                </div>
                <!-- Last Frame -->
                <div class="flex-1 flex flex-col items-center gap-1.5">
                  <div class="w-full aspect-video rounded-lg border-2 border-dashed border-[--cinema-gold]/20 bg-[--cinema-panel]/50 flex items-center justify-center cursor-pointer hover:border-[--cinema-gold]/40 transition-colors">
                    <div class="flex flex-col items-center gap-1">
                      <span class="material-symbols-outlined text-[--cinema-gold]/20 text-2xl">add_photo_alternate</span>
                      <span class="text-[9px] text-[--cinema-text-dim]">Upload or generate</span>
                    </div>
                  </div>
                  <span class="text-[10px] text-[--cinema-text-muted] font-mono uppercase">Last Frame</span>
                </div>
              </div>
            </div>

            <div class="gold-line mb-6"></div>
          }

          <!-- Parameters -->
          <div class="mb-6">
            <h4 class="font-display uppercase text-xs text-[--cinema-gold]/60 tracking-wider mb-4">Parameters</h4>

            <!-- Duration -->
            <div class="mb-5">
              <div class="flex items-center justify-between mb-2">
                <label class="text-xs text-[--cinema-text-muted]">Duration</label>
                <span class="text-xs text-[--cinema-gold] font-mono">{{ duration() }}s</span>
              </div>
              <input
                type="range"
                [min]="1" [max]="10" [step]="0.5"
                [ngModel]="duration()"
                (ngModelChange)="duration.set($event)"
                class="w-full slider-gold cursor-pointer"
              />
              <div class="flex justify-between text-[10px] text-[--cinema-text-dim] mt-1">
                <span>1s</span>
                <span>10s</span>
              </div>
            </div>

            <!-- Motion Strength -->
            <div>
              <div class="flex items-center justify-between mb-2">
                <label class="text-xs text-[--cinema-text-muted]">Motion Strength</label>
                <span class="text-xs text-[--cinema-gold] font-mono">{{ motionStrength() }}</span>
              </div>
              <input
                type="range"
                [min]="1" [max]="10" [step]="1"
                [ngModel]="motionStrength()"
                (ngModelChange)="motionStrength.set($event)"
                class="w-full slider-gold cursor-pointer"
              />
              <div class="flex justify-between text-[10px] text-[--cinema-text-dim] mt-1">
                <span>Subtle</span>
                <span>Dynamic</span>
              </div>
            </div>
          </div>

          <div class="gold-line mb-6"></div>

          <!-- Process Film Button -->
          <button
            [disabled]="!scene()?.confirmed_image_url"
            class="btn-velvet w-full py-4 font-cinema text-lg rounded-xl flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <span class="material-symbols-outlined text-[22px]">local_fire_department</span>
            Process Film
          </button>

          <!-- Output Preview -->
          @if (scene()?.generated_video_url) {
            <div class="mt-6">
              <h4 class="font-display uppercase text-xs text-[--cinema-gold]/60 tracking-wider mb-3">Output Preview</h4>
              <div class="relative aspect-video rounded-lg brass-frame overflow-hidden group cursor-pointer bg-[--cinema-panel]">
                @if (scene()!.video_thumbnail_url) {
                  <img [src]="scene()!.video_thumbnail_url!" class="w-full h-full object-cover" alt="Video output" />
                } @else {
                  <div class="absolute inset-0 bg-gradient-to-br from-[--cinema-surface] to-[--cinema-panel] flex items-center justify-center">
                    <span class="material-symbols-outlined text-[--cinema-gold]/20 text-4xl">movie</span>
                  </div>
                }
                <div class="absolute inset-0 flex items-center justify-center bg-black/20 group-hover:bg-black/40 transition-colors">
                  <div class="w-14 h-14 rounded-full bg-black/50 flex items-center justify-center backdrop-blur-sm group-hover:scale-110 transition-transform border border-[--cinema-gold]/20">
                    <span class="material-symbols-outlined text-[--cinema-gold] text-[28px] fill-1">play_arrow</span>
                  </div>
                </div>
              </div>
              <p class="text-green-400 text-sm mt-2">Film processed successfully</p>
            </div>
          }

        </div>
      </div>

      <!-- Bottom Status Bar -->
      <footer class="h-10 flex-shrink-0 flex items-center px-6 border-t border-[--cinema-gold]/10 bg-[--cinema-bg]">
        @if (scene()?.generated_video_url) {
          <div class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-green-400"></span>
            <span class="text-xs text-[--cinema-text-muted]">Film processed and ready for review</span>
          </div>
        } @else if (scene()?.confirmed_image_url) {
          <div class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-amber-400"></span>
            <span class="text-xs text-[--cinema-text-muted]">Source frame loaded, awaiting processing</span>
          </div>
        } @else {
          <div class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-[--cinema-text-dim]"></span>
            <span class="text-xs text-[--cinema-text-muted]">No source frame available</span>
          </div>
        }
      </footer>

    </div>
  `,
  styles: [`
    .no-scrollbar::-webkit-scrollbar { display: none; }
    .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
    .slider-gold {
      -webkit-appearance: none;
      appearance: none;
      height: 4px;
      background: var(--cinema-border);
      border-radius: 2px;
      outline: none;
    }
    .slider-gold::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--cinema-gold);
      cursor: pointer;
      border: 2px solid var(--cinema-bg);
    }
    .slider-gold::-moz-range-thumb {
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--cinema-gold);
      cursor: pointer;
      border: 2px solid var(--cinema-bg);
    }
  `]
})
export class DirectorsCutVideogenComponent {
  private vellumService = inject(VellumiumService);

  scene = computed(() => this.vellumService.selectedScene());

  method = signal<I2vMethod>('i2v');
  duration = signal(4);
  motionStrength = signal(5);

  onBack() {
    this.vellumService.backToEditor();
  }
}
