
import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { VellumiumService } from '../../services/vellumium.service';
import { CanvasService } from '../../services/canvas.service';
import { CanvasViewportComponent } from './canvas-viewport.component';
import { CanvasToolbarComponent } from './canvas-toolbar.component';
import { CanvasToolOptionsComponent } from './canvas-tool-options.component';
import { CanvasLayersPanelComponent } from './canvas-layers-panel.component';
import { CanvasGenerationPanelComponent } from './canvas-generation-panel.component';

@Component({
  selector: 'app-storyboard-canvas',
  standalone: true,
  imports: [
    CommonModule, FormsModule,
    CanvasViewportComponent,
    CanvasToolbarComponent,
    CanvasToolOptionsComponent,
    CanvasLayersPanelComponent,
    CanvasGenerationPanelComponent,
  ],
  template: `
    <div class="h-screen flex flex-col bg-[--cinema-bg] font-display overflow-hidden">

      <!-- Header -->
      <header class="flex items-center justify-between whitespace-nowrap border-b border-[--cinema-border] bg-[--cinema-bg] px-4 py-2 z-50 shrink-0">
        <div class="flex items-center gap-3">
          <button class="p-1.5 rounded-lg text-[--cinema-text-muted] hover:text-[--cinema-primary] hover:bg-[--cinema-primary]/5 transition-colors" (click)="vellumService.backToEditor()">
            <span class="material-symbols-outlined text-[18px]">arrow_back</span>
          </button>
          <div class="w-px h-5 bg-[--cinema-border]"></div>
          <h2 class="text-[--cinema-text] text-sm font-medium">{{ vellumService.selectedScene()?.title }}</h2>
          @if (vellumService.selectedScene()?.description) {
            <span class="text-[--cinema-text-dim] text-xs hidden md:inline">{{ vellumService.selectedScene()?.description }}</span>
          }
        </div>

        <div class="flex items-center gap-2">
          <!-- Zoom display -->
          <span class="text-[10px] text-[--cinema-text-muted] font-mono">{{ (canvasService.zoom() * 100).toFixed(0) }}%</span>

          <!-- Right panel toggle -->
          <button
            class="p-1.5 rounded-lg transition-colors"
            [class]="showRightPanel()
              ? 'text-[--cinema-primary] bg-[--cinema-primary]/10'
              : 'text-[--cinema-text-muted] hover:text-[--cinema-primary] hover:bg-[--cinema-primary]/5'"
            (click)="toggleRightPanel()"
            title="Toggle Panel">
            <span class="material-symbols-outlined text-[18px]">side_navigation</span>
          </button>

          <!-- Right tab switcher -->
          @if (showRightPanel()) {
            <div class="flex bg-[--cinema-panel] rounded-lg p-0.5 border border-[--cinema-border]">
              <button
                class="px-3 py-1 rounded-md text-[10px] font-display font-medium transition-all"
                [class]="rightTab() === 'generate'
                  ? 'bg-[--cinema-primary]/15 text-[--cinema-primary]'
                  : 'text-[--cinema-text-muted] hover:text-[--cinema-primary]'"
                (click)="rightTab.set('generate')">Generate</button>
              <button
                class="px-3 py-1 rounded-md text-[10px] font-display font-medium transition-all"
                [class]="rightTab() === 'layers'
                  ? 'bg-[--cinema-primary]/15 text-[--cinema-primary]'
                  : 'text-[--cinema-text-muted] hover:text-[--cinema-primary]'"
                (click)="rightTab.set('layers')">Layers</button>
            </div>
          }
        </div>
      </header>

      <!-- Tool Options Bar -->
      <app-canvas-tool-options></app-canvas-tool-options>

      <!-- Main Layout: Toolbar | Viewport | Right Panel -->
      <main class="flex-1 flex overflow-hidden relative">

        <!-- Left Toolbar -->
        <app-canvas-toolbar></app-canvas-toolbar>

        <!-- Canvas Viewport -->
        <section class="flex-1 relative overflow-hidden">
          <app-canvas-viewport></app-canvas-viewport>
        </section>

        <!-- Right Panel -->
        @if (showRightPanel()) {
          <aside class="w-80 bg-[--cinema-surface] border-l border-[--cinema-border] flex flex-col z-40 shrink-0">
            @if (rightTab() === 'generate') {
              <app-canvas-generation-panel></app-canvas-generation-panel>
            } @else {
              <app-canvas-layers-panel></app-canvas-layers-panel>
            }
          </aside>
        }

      </main>
    </div>
  `
})
export class StoryboardCanvasComponent {
  vellumService = inject(VellumiumService);
  canvasService = inject(CanvasService);

  rightTab = signal<'generate' | 'layers'>('generate');
  showRightPanel = signal(true);

  toggleRightPanel() {
    this.showRightPanel.set(!this.showRightPanel());
  }
}
