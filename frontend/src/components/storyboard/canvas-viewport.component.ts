
import { Component, ElementRef, ViewChild, AfterViewInit, OnDestroy, inject, HostListener, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CanvasService } from '../../services/canvas.service';
import { VellumiumService } from '../../services/vellumium.service';

@Component({
  selector: 'app-canvas-viewport',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div
      class="relative w-full h-full overflow-hidden"
      style="background-image: url('data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2220%22 height=%2220%22><rect width=%2210%22 height=%2210%22 fill=%22%231a1a2e%22/><rect x=%2210%22 y=%2210%22 width=%2210%22 height=%2210%22 fill=%22%231a1a2e%22/><rect x=%2210%22 width=%2210%22 height=%2210%22 fill=%22%23161628%22/><rect y=%2210%22 width=%2210%22 height=%2210%22 fill=%22%23161628%22/></svg>'); background-size: 20px 20px;"
      (wheel)="onWheel($event)"
      (dragover)="onDragOver($event)"
      (drop)="onDrop($event)"
    >
      <canvas #fabricCanvas></canvas>

      <!-- Frame / crop overlay -->
      @if (canvasService.frameEnabled()) {
        <div class="absolute inset-0 pointer-events-none z-10">
          <!-- Top dim -->
          <div class="absolute bg-black/60" [style]="frameDimStyle('top')"></div>
          <!-- Bottom dim -->
          <div class="absolute bg-black/60" [style]="frameDimStyle('bottom')"></div>
          <!-- Left dim -->
          <div class="absolute bg-black/60" [style]="frameDimStyle('left')"></div>
          <!-- Right dim -->
          <div class="absolute bg-black/60" [style]="frameDimStyle('right')"></div>
          <!-- Frame border -->
          <div class="absolute border border-white/60" [style]="frameBorderStyle()"></div>
          <!-- Corner accents -->
          <div class="absolute border-t-2 border-l-2 border-white/90 w-3 h-3" [style]="frameCornerStyle('tl')"></div>
          <div class="absolute border-t-2 border-r-2 border-white/90 w-3 h-3" [style]="frameCornerStyle('tr')"></div>
          <div class="absolute border-b-2 border-l-2 border-white/90 w-3 h-3" [style]="frameCornerStyle('bl')"></div>
          <div class="absolute border-b-2 border-r-2 border-white/90 w-3 h-3" [style]="frameCornerStyle('br')"></div>
          <!-- Aspect ratio label -->
          <div class="absolute text-[10px] text-white/50 font-display px-1.5 py-0.5 bg-black/40 rounded-sm" [style]="frameLabelStyle()">
            {{ canvasService.frameAspectRatio() }} &middot; {{ canvasService.frameRect().w }}&times;{{ canvasService.frameRect().h }}
          </div>
        </div>
      }

      <!-- Brush cursor overlay (follows mouse for brush/eraser/mask) -->
      @if (canvasService.brushCursor().visible) {
        <div
          class="absolute pointer-events-none rounded-full"
          [style.left.px]="canvasService.brushCursor().x - canvasService.brushCursor().size / 2"
          [style.top.px]="canvasService.brushCursor().y - canvasService.brushCursor().size / 2"
          [style.width.px]="canvasService.brushCursor().size"
          [style.height.px]="canvasService.brushCursor().size"
          [style.borderWidth.px]="1.5"
          [style.borderStyle]="cursorBorderStyle()"
          [style.borderColor]="cursorColor()"
          [style.boxShadow]="'0 0 3px rgba(0,0,0,0.5)'"
        ></div>
      }

      <!-- Zoom indicator -->
      <div class="absolute bottom-4 right-4 bg-[--cinema-surface]/80 backdrop-blur-sm text-[--cinema-text-muted] text-xs px-3 py-1.5 rounded-full border border-[--cinema-border] font-display">
        {{ (canvasService.zoom() * 100).toFixed(0) }}%
      </div>

      <!-- Crosshair guides -->
      <div class="absolute inset-0 pointer-events-none opacity-10">
        <div class="absolute top-1/2 left-0 w-full h-px bg-[--cinema-primary]/30"></div>
        <div class="absolute left-1/2 top-0 h-full w-px bg-[--cinema-primary]/30"></div>
      </div>

      <!-- Drop zone overlay -->
      @if (isDragOver) {
        <div class="absolute inset-0 bg-[--cinema-primary]/10 border-2 border-dashed border-[--cinema-primary] flex items-center justify-center z-50">
          <div class="text-[--cinema-primary] font-display text-sm flex items-center gap-2">
            <span class="material-symbols-outlined">add_photo_alternate</span>
            Drop image to add as layer
          </div>
        </div>
      }
    </div>
  `
})
export class CanvasViewportComponent implements AfterViewInit, OnDestroy {
  @ViewChild('fabricCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;

  canvasService = inject(CanvasService);
  private vellumService = inject(VellumiumService);
  isDragOver = false;
  private resizeObserver: ResizeObserver | null = null;

  private autoSaveInterval: ReturnType<typeof setInterval> | null = null;

  // Brush cursor appearance based on active tool
  cursorColor = computed(() => {
    const tool = this.canvasService.activeTool();
    if (tool === 'eraser') return 'rgba(255,100,100,0.8)';
    if (tool === 'mask') {
      return this.canvasService.maskColor() === 'white'
        ? 'rgba(255,60,60,0.8)'
        : 'rgba(60,120,255,0.8)';
    }
    return 'rgba(255,255,255,0.8)'; // brush
  });

  cursorBorderStyle = computed(() => {
    const tool = this.canvasService.activeTool();
    return tool === 'eraser' ? 'dashed' : 'solid';
  });

  // ─── Frame overlay style helpers ─────────────────────────
  frameDimStyle(side: 'top' | 'bottom' | 'left' | 'right'): string {
    const { fx, fy, fw, fh } = this.canvasService.getFrameScreenRect();
    switch (side) {
      case 'top':    return `left:0;top:0;right:0;height:${Math.max(0, fy)}px`;
      case 'bottom': return `left:0;top:${fy + fh}px;right:0;bottom:0`;
      case 'left':   return `left:0;top:${fy}px;width:${Math.max(0, fx)}px;height:${fh}px`;
      case 'right':  return `left:${fx + fw}px;top:${fy}px;right:0;height:${fh}px`;
    }
  }

  frameBorderStyle(): string {
    const { fx, fy, fw, fh } = this.canvasService.getFrameScreenRect();
    return `left:${fx}px;top:${fy}px;width:${fw}px;height:${fh}px`;
  }

  frameCornerStyle(corner: 'tl' | 'tr' | 'bl' | 'br'): string {
    const { fx, fy, fw, fh } = this.canvasService.getFrameScreenRect();
    switch (corner) {
      case 'tl': return `left:${fx - 1}px;top:${fy - 1}px`;
      case 'tr': return `left:${fx + fw - 11}px;top:${fy - 1}px`;
      case 'bl': return `left:${fx - 1}px;top:${fy + fh - 11}px`;
      case 'br': return `left:${fx + fw - 11}px;top:${fy + fh - 11}px`;
    }
  }

  frameLabelStyle(): string {
    const { fx, fy, fw } = this.canvasService.getFrameScreenRect();
    return `left:${fx + fw / 2}px;top:${fy - 22}px;transform:translateX(-50%)`;
  }

  async ngAfterViewInit() {
    const el = this.canvasRef.nativeElement;
    const parent = el.parentElement!;
    const w = parent.clientWidth;
    const h = parent.clientHeight;

    this.canvasService.initCanvas(el, w, h);

    // Position frame overlay centered on the canvas
    this.canvasService.centerFrame();

    // Try to load draft first, otherwise load scene thumbnail
    const scene = this.vellumService.selectedScene();
    const draftLoaded = scene ? await this.canvasService.loadDraft(scene.id) : false;

    if (!draftLoaded) {
      if (scene?.thumbnail_url) {
        this.canvasService.addImageLayer('Base Image', scene.thumbnail_url);
      } else {
        this.canvasService.addLayer('Base', 'draw');
      }
    }

    // Auto-save draft every 30 seconds
    if (scene) {
      this.autoSaveInterval = setInterval(() => {
        this.canvasService.saveDraft(scene.id);
      }, 30000);
    }

    // Observe resize
    this.resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (this.canvasService.canvas) {
          this.canvasService.canvas.setDimensions({ width, height });
          this.canvasService.canvas.requestRenderAll();
          this.canvasService.centerFrame();
        }
      }
    });
    this.resizeObserver.observe(parent);
  }

  ngOnDestroy() {
    // Save draft on exit
    const scene = this.vellumService.selectedScene();
    if (scene) {
      this.canvasService.saveDraft(scene.id);
    }
    if (this.autoSaveInterval) clearInterval(this.autoSaveInterval);
    this.resizeObserver?.disconnect();
    this.canvasService.dispose();
  }

  onWheel(e: WheelEvent) {
    this.canvasService.handleWheel(e);
  }

  onDragOver(e: DragEvent) {
    e.preventDefault();
    this.isDragOver = true;
  }

  onDrop(e: DragEvent) {
    e.preventDefault();
    this.isDragOver = false;
    const files = e.dataTransfer?.files;
    if (files) {
      for (let i = 0; i < files.length; i++) {
        if (files[i].type.startsWith('image/')) {
          this.canvasService.handleDroppedFile(files[i]);
        }
      }
    }
  }

  @HostListener('dragleave')
  onDragLeave() {
    this.isDragOver = false;
  }

  @HostListener('document:keydown', ['$event'])
  onKeyDown(e: KeyboardEvent) {
    // Skip all shortcuts when typing in input/textarea
    const isTyping = e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement
      || (e.target as HTMLElement)?.isContentEditable;
    if (e.code === 'Space' && !e.repeat && !isTyping) {
      e.preventDefault();
      this.canvasService.setSpaceHeld(true);
    }
    if (isTyping) return;

    if (e.ctrlKey && e.key === 'z' && !e.shiftKey) { e.preventDefault(); this.canvasService.undo(); }
    if (e.ctrlKey && e.key === 'z' && e.shiftKey) { e.preventDefault(); this.canvasService.redo(); }
    if (e.ctrlKey && e.key === 'y') { e.preventDefault(); this.canvasService.redo(); }

    if (!e.ctrlKey && !e.altKey) {
      switch (e.key.toLowerCase()) {
        case 'v': this.canvasService.setTool('select'); break;
        case 'h': this.canvasService.setTool('pan'); break;
        case 'b': this.canvasService.setTool('brush'); break;
        case 'e': this.canvasService.setTool('eraser'); break;
        case 'g': this.canvasService.setTool('fill'); break;
        case 'l': this.canvasService.setTool('line'); break;
        case 'm': this.canvasService.setTool('mask'); break;
        case 'r': this.canvasService.setTool('rect_select'); break;
        case 'z': this.canvasService.setTool('zoom'); break;
      }
    }

    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (this.canvasService.hasSelection()) {
        this.canvasService.deleteWithinSelection();
      }
    }

    // Escape clears selection
    if (e.key === 'Escape') {
      this.canvasService.clearSelection();
    }
  }

  @HostListener('document:keyup', ['$event'])
  onKeyUp(e: KeyboardEvent) {
    if (e.code === 'Space') {
      this.canvasService.setSpaceHeld(false);
    }
  }
}
