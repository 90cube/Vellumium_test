
import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { CanvasService, CanvasLayer } from '../../services/canvas.service';

@Component({
  selector: 'app-canvas-layers-panel',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="flex flex-col h-full">
      <!-- Header -->
      <div class="flex items-center justify-between px-4 py-3 border-b border-[--cinema-border]">
        <h3 class="text-xs font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Layers</h3>
        <div class="flex gap-1">
          <button (click)="imageFileInput.click()" class="p-1 rounded hover:bg-[--cinema-primary]/10 text-[--cinema-text-muted] hover:text-[--cinema-primary] transition-colors" title="Upload Image">
            <span class="material-symbols-outlined text-[16px]">add_photo_alternate</span>
          </button>
          <button (click)="addDrawLayer()" class="p-1 rounded hover:bg-[--cinema-primary]/10 text-[--cinema-text-muted] hover:text-[--cinema-primary] transition-colors" title="Add Draw Layer">
            <span class="material-symbols-outlined text-[16px]">brush</span>
          </button>
          <button (click)="addMaskLayer()" class="p-1 rounded hover:bg-[--cinema-primary]/10 text-[--cinema-text-muted] hover:text-[--cinema-primary] transition-colors" title="Add Mask Layer">
            <span class="material-symbols-outlined text-[16px]">deblur</span>
          </button>
          <button (click)="addTextLayer()" class="p-1 rounded hover:bg-[--cinema-primary]/10 text-[--cinema-text-muted] hover:text-[--cinema-primary] transition-colors" title="Add Text Layer">
            <span class="material-symbols-outlined text-[16px]">text_fields</span>
          </button>
          <div class="w-px h-4 bg-[--cinema-border] mx-0.5"></div>
          <button (click)="cleanupOrphans()" class="p-1 rounded hover:bg-[--cinema-primary]/10 text-[--cinema-text-muted] hover:text-[--cinema-primary] transition-colors" title="Remove orphaned objects">
            <span class="material-symbols-outlined text-[16px]">mop</span>
          </button>
          <button (click)="clearAll()" class="p-1 rounded hover:bg-[--cinema-error]/10 text-[--cinema-text-muted] hover:text-[--cinema-error] transition-colors" title="Clear All">
            <span class="material-symbols-outlined text-[16px]">delete_sweep</span>
          </button>
        </div>
        <input #imageFileInput type="file" accept="image/*" multiple class="hidden" (change)="onImageFilesSelected($event)">
      </div>

      <!-- Layer List (reversed so top layer appears first) -->
      <div class="flex-1 overflow-y-auto p-2 space-y-1">
        @for (layer of reversedLayers(); track layer.id; let i = $index) {
          <div
            class="group flex items-center gap-2 p-2 rounded-lg border transition-all cursor-pointer"
            [class]="canvasService.activeLayerId() === layer.id
              ? 'bg-[--cinema-primary]/10 border-[--cinema-primary]/30'
              : 'bg-[--cinema-surface] border-[--cinema-border] hover:border-[--cinema-primary]/20'"
            (click)="canvasService.setActiveLayer(layer.id)"
            draggable="true"
            (dragstart)="onDragStart($event, layer)"
            (dragover)="onDragOver($event)"
            (drop)="onDrop($event, layer)"
          >
            <!-- Drag handle -->
            <span class="material-symbols-outlined text-[14px] text-[--cinema-text-dim] cursor-grab group-hover:text-[--cinema-text-muted]">drag_indicator</span>

            <!-- Layer type icon -->
            @switch (layer.type) {
              @case ('image') {
                <span class="material-symbols-outlined text-[16px] text-[--cinema-primary]/60">image</span>
              }
              @case ('draw') {
                <span class="material-symbols-outlined text-[16px] text-green-400/60">brush</span>
              }
              @case ('mask') {
                <span class="material-symbols-outlined text-[16px] text-red-400/60">deblur</span>
              }
              @case ('text') {
                <span class="material-symbols-outlined text-[16px] text-yellow-400/60">text_fields</span>
              }
            }

            <!-- Name -->
            @if (editingLayerId === layer.id) {
              <input
                class="flex-1 bg-transparent text-[--cinema-text] text-xs border-b border-[--cinema-primary] outline-none px-1 py-0.5"
                [value]="layer.name"
                (blur)="finishRename($event, layer.id)"
                (keydown.enter)="finishRename($event, layer.id)"
                #renameInput
              >
            } @else {
              <span
                class="flex-1 text-xs text-[--cinema-text] truncate"
                (dblclick)="startRename(layer.id)"
              >{{ layer.name }}</span>
            }

            <!-- Controls -->
            <div class="flex items-center gap-0.5 opacity-60 group-hover:opacity-100 transition-opacity">
              <!-- Visibility -->
              <button
                class="p-0.5 rounded hover:bg-white/5 transition-colors"
                [class.text-[--cinema-text-muted]]="layer.visible"
                [class.text-[--cinema-error]/50]="!layer.visible"
                (click)="toggleVisibility($event, layer)"
                [title]="layer.visible ? 'Hide' : 'Show'">
                <span class="material-symbols-outlined text-[14px]">{{ layer.visible ? 'visibility' : 'visibility_off' }}</span>
              </button>

              <!-- Lock -->
              <button
                class="p-0.5 rounded hover:bg-white/5 transition-colors"
                [class.text-[--cinema-error]/50]="layer.locked"
                [class.text-[--cinema-text-muted]]="!layer.locked"
                (click)="toggleLock($event, layer)"
                [title]="layer.locked ? 'Unlock' : 'Lock'">
                <span class="material-symbols-outlined text-[14px]">{{ layer.locked ? 'lock' : 'lock_open' }}</span>
              </button>
            </div>
          </div>
        }

        @if (canvasService.layers().length === 0) {
          <div class="text-xs text-[--cinema-text-dim] text-center py-6 font-display">No layers yet</div>
        }
      </div>

      <!-- Layer Properties (active layer) -->
      @if (canvasService.activeLayer(); as layer) {
        <div class="border-t border-[--cinema-border] p-3 space-y-3">
          <!-- Opacity -->
          <div class="flex items-center gap-2">
            <span class="text-[10px] text-[--cinema-text-muted] uppercase tracking-wider w-14">Opacity</span>
            <input type="range" min="0" max="100"
              [value]="layer.opacity * 100"
              (input)="canvasService.setLayerOpacity(layer.id, +$any($event.target).value / 100)"
              class="flex-1 h-1 accent-[--cinema-primary] bg-[--cinema-border] rounded-full cursor-pointer">
            <span class="text-[10px] text-[--cinema-text] w-8 text-right">{{ (layer.opacity * 100).toFixed(0) }}%</span>
          </div>

          <!-- Blend Mode -->
          <div class="flex items-center gap-2">
            <span class="text-[10px] text-[--cinema-text-muted] uppercase tracking-wider w-14">Blend</span>
            <select
              class="flex-1 bg-[--cinema-bg] text-[--cinema-text] text-[10px] border border-[--cinema-border] rounded px-2 py-1 outline-none focus:border-[--cinema-primary]/30"
              [value]="layer.blendMode"
              (change)="canvasService.setLayerBlendMode(layer.id, $any($event.target).value)">
              @for (mode of blendModes; track mode.value) {
                <option [value]="mode.value">{{ mode.label }}</option>
              }
            </select>
          </div>

          <!-- Actions -->
          <div class="flex gap-1">
            <button class="flex-1 px-2 py-1 text-[10px] rounded bg-[--cinema-panel] text-[--cinema-text-muted] border border-[--cinema-border] hover:text-[--cinema-primary] hover:border-[--cinema-primary]/30 transition-colors"
              (click)="canvasService.duplicateLayer(layer.id)">Duplicate</button>
            <button class="flex-1 px-2 py-1 text-[10px] rounded bg-[--cinema-panel] text-[--cinema-text-muted] border border-[--cinema-border] hover:text-[--cinema-primary] hover:border-[--cinema-primary]/30 transition-colors"
              (click)="canvasService.mergeDown(layer.id)">Merge Down</button>
            <button class="px-2 py-1 text-[10px] rounded bg-[--cinema-panel] text-[--cinema-text-muted] border border-[--cinema-border] hover:text-[--cinema-error] hover:border-[--cinema-error]/30 transition-colors"
              (click)="canvasService.removeLayer(layer.id)">
              <span class="material-symbols-outlined text-[12px]">delete</span>
            </button>
          </div>
        </div>
      }
    </div>
  `
})
export class CanvasLayersPanelComponent {
  canvasService = inject(CanvasService);
  editingLayerId: string | null = null;
  private dragLayerId: string | null = null;

  blendModes = [
    { value: 'source-over', label: 'Normal' },
    { value: 'multiply', label: 'Multiply' },
    { value: 'screen', label: 'Screen' },
    { value: 'overlay', label: 'Overlay' },
    { value: 'darken', label: 'Darken' },
    { value: 'lighten', label: 'Lighten' },
    { value: 'color-dodge', label: 'Color Dodge' },
    { value: 'color-burn', label: 'Color Burn' },
    { value: 'hard-light', label: 'Hard Light' },
    { value: 'soft-light', label: 'Soft Light' },
    { value: 'difference', label: 'Difference' },
    { value: 'exclusion', label: 'Exclusion' },
    { value: 'hue', label: 'Hue' },
    { value: 'saturation', label: 'Saturation' },
    { value: 'color', label: 'Color' },
    { value: 'luminosity', label: 'Luminosity' },
  ];

  reversedLayers() {
    return [...this.canvasService.layers()].reverse();
  }

  onImageFilesSelected(e: Event) {
    const input = e.target as HTMLInputElement;
    if (!input.files) return;
    for (let i = 0; i < input.files.length; i++) {
      const file = input.files[i];
      if (file.type.startsWith('image/')) {
        this.canvasService.handleDroppedFile(file);
      }
    }
    input.value = '';
  }

  addDrawLayer() {
    this.canvasService.addLayer('Draw ' + (this.canvasService.layers().length + 1), 'draw');
  }

  addMaskLayer() {
    this.canvasService.addLayer('Mask', 'mask');
  }

  addTextLayer() {
    this.canvasService.addLayer('Text ' + (this.canvasService.layers().length + 1), 'text');
  }

  cleanupOrphans() {
    this.canvasService.removeOrphanedObjects();
  }

  clearAll() {
    this.canvasService.clearAll();
  }

  toggleVisibility(e: Event, layer: CanvasLayer) {
    e.stopPropagation();
    this.canvasService.setLayerVisibility(layer.id, !layer.visible);
  }

  toggleLock(e: Event, layer: CanvasLayer) {
    e.stopPropagation();
    this.canvasService.setLayerLocked(layer.id, !layer.locked);
  }

  startRename(layerId: string) {
    this.editingLayerId = layerId;
  }

  finishRename(e: Event, layerId: string) {
    const input = e.target as HTMLInputElement;
    if (input.value.trim()) {
      this.canvasService.renameLayer(layerId, input.value.trim());
    }
    this.editingLayerId = null;
  }

  onDragStart(e: DragEvent, layer: CanvasLayer) {
    this.dragLayerId = layer.id;
    e.dataTransfer!.effectAllowed = 'move';
  }

  onDragOver(e: DragEvent) {
    e.preventDefault();
    e.dataTransfer!.dropEffect = 'move';
  }

  onDrop(e: DragEvent, targetLayer: CanvasLayer) {
    e.preventDefault();
    if (!this.dragLayerId || this.dragLayerId === targetLayer.id) return;

    const layers = this.canvasService.layers();
    const fromIdx = layers.findIndex(l => l.id === this.dragLayerId);
    const toIdx = layers.findIndex(l => l.id === targetLayer.id);
    if (fromIdx >= 0 && toIdx >= 0) {
      this.canvasService.reorderLayers(fromIdx, toIdx);
    }
    this.dragLayerId = null;
  }
}
