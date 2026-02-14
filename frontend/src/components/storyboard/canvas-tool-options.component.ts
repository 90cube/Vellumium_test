
import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { CanvasService } from '../../services/canvas.service';

@Component({
  selector: 'app-canvas-tool-options',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="h-10 bg-[--cinema-surface] border-b border-[--cinema-border] flex items-center px-4 gap-4 text-xs font-display shrink-0">

      <!-- Tool name -->
      <span class="text-[--cinema-text-muted] uppercase tracking-wider font-semibold min-w-[80px]">
        {{ toolDisplayName() }}
      </span>

      <div class="w-px h-5 bg-[--cinema-border]"></div>

      @switch (canvasService.activeTool()) {

        @case ('brush') {
          <!-- Brush Size -->
          <label class="flex items-center gap-2 text-[--cinema-text-muted]">
            Size
            <input type="range" min="1" max="100" [value]="canvasService.brushOptions().size"
              (input)="canvasService.setBrushSize(+$any($event.target).value)"
              class="w-20 h-1 accent-[--cinema-primary] bg-[--cinema-border] rounded-full cursor-pointer">
            <span class="text-[--cinema-text] w-6 text-right">{{ canvasService.brushOptions().size }}</span>
          </label>

          <!-- Color -->
          <label class="flex items-center gap-2 text-[--cinema-text-muted]">
            Color
            <input type="color" [value]="canvasService.brushOptions().color"
              (input)="canvasService.setBrushColor($any($event.target).value)"
              class="w-6 h-6 rounded border border-[--cinema-border] cursor-pointer bg-transparent">
          </label>

          <!-- Opacity -->
          <label class="flex items-center gap-2 text-[--cinema-text-muted]">
            Opacity
            <input type="range" min="0" max="100" [value]="canvasService.brushOptions().opacity * 100"
              (input)="canvasService.setBrushOpacity(+$any($event.target).value / 100)"
              class="w-16 h-1 accent-[--cinema-primary] bg-[--cinema-border] rounded-full cursor-pointer">
            <span class="text-[--cinema-text] w-8 text-right">{{ (canvasService.brushOptions().opacity * 100).toFixed(0) }}%</span>
          </label>
        }

        @case ('eraser') {
          <label class="flex items-center gap-2 text-[--cinema-text-muted]">
            Size
            <input type="range" min="1" max="100" [value]="canvasService.brushOptions().size"
              (input)="canvasService.setBrushSize(+$any($event.target).value)"
              class="w-20 h-1 accent-[--cinema-primary] bg-[--cinema-border] rounded-full cursor-pointer">
            <span class="text-[--cinema-text] w-6 text-right">{{ canvasService.brushOptions().size }}</span>
          </label>

          <label class="flex items-center gap-2 text-[--cinema-text-muted]">
            Opacity
            <input type="range" min="0" max="100" [value]="canvasService.brushOptions().opacity * 100"
              (input)="canvasService.setBrushOpacity(+$any($event.target).value / 100)"
              class="w-16 h-1 accent-[--cinema-primary] bg-[--cinema-border] rounded-full cursor-pointer">
            <span class="text-[--cinema-text] w-8 text-right">{{ (canvasService.brushOptions().opacity * 100).toFixed(0) }}%</span>
          </label>
        }

        @case ('line') {
          <label class="flex items-center gap-2 text-[--cinema-text-muted]">
            Width
            <input type="range" min="1" max="50" [value]="canvasService.brushOptions().size"
              (input)="canvasService.setBrushSize(+$any($event.target).value)"
              class="w-20 h-1 accent-[--cinema-primary] bg-[--cinema-border] rounded-full cursor-pointer">
            <span class="text-[--cinema-text] w-6 text-right">{{ canvasService.brushOptions().size }}</span>
          </label>
          <label class="flex items-center gap-2 text-[--cinema-text-muted]">
            Color
            <input type="color" [value]="canvasService.brushOptions().color"
              (input)="canvasService.setBrushColor($any($event.target).value)"
              class="w-6 h-6 rounded border border-[--cinema-border] cursor-pointer bg-transparent">
          </label>
        }

        @case ('fill') {
          <label class="flex items-center gap-2 text-[--cinema-text-muted]">
            Color
            <input type="color" [value]="canvasService.brushOptions().color"
              (input)="canvasService.setBrushColor($any($event.target).value)"
              class="w-6 h-6 rounded border border-[--cinema-border] cursor-pointer bg-transparent">
          </label>
          <label class="flex items-center gap-2 text-[--cinema-text-muted]">
            Tolerance
            <input type="range" min="0" max="128" [value]="canvasService.fillTolerance()"
              (input)="canvasService.fillTolerance.set(+$any($event.target).value)"
              class="w-20 h-1 accent-[--cinema-primary] bg-[--cinema-border] rounded-full cursor-pointer">
            <span class="text-[--cinema-text] w-6 text-right">{{ canvasService.fillTolerance() }}</span>
          </label>
        }

        @case ('mask') {
          <label class="flex items-center gap-2 text-[--cinema-text-muted]">
            Size
            <input type="range" min="1" max="100" [value]="canvasService.brushOptions().size"
              (input)="canvasService.setBrushSize(+$any($event.target).value)"
              class="w-20 h-1 accent-[--cinema-primary] bg-[--cinema-border] rounded-full cursor-pointer">
            <span class="text-[--cinema-text] w-6 text-right">{{ canvasService.brushOptions().size }}</span>
          </label>

          <div class="flex items-center gap-1.5 text-[--cinema-text-muted]">
            Mode
            <button
              class="px-2 py-0.5 rounded text-xs transition-colors"
              [class]="canvasService.maskColor() === 'white' ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 'bg-[--cinema-panel] text-[--cinema-text-muted] border border-[--cinema-border] hover:border-red-500/30'"
              (click)="canvasService.setMaskColor('white')">
              Edit
            </button>
            <button
              class="px-2 py-0.5 rounded text-xs transition-colors"
              [class]="canvasService.maskColor() === 'black' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' : 'bg-[--cinema-panel] text-[--cinema-text-muted] border border-[--cinema-border] hover:border-blue-500/30'"
              (click)="canvasService.setMaskColor('black')">
              Keep
            </button>
          </div>

          <span class="text-[--cinema-text-dim]">Alt+Click to toggle</span>
        }

        @case ('rect_select') {
          @if (canvasService.hasSelection()) {
            <ng-container *ngTemplateOutlet="selectionActions"></ng-container>
          } @else {
            <span class="text-[--cinema-text-dim]">Click and drag to select area</span>
          }
        }

        @case ('ellipse_select') {
          @if (canvasService.hasSelection()) {
            <ng-container *ngTemplateOutlet="selectionActions"></ng-container>
          } @else {
            <span class="text-[--cinema-text-dim]">Click and drag to select area</span>
          }
        }

        @case ('lasso_select') {
          @if (canvasService.hasSelection()) {
            <ng-container *ngTemplateOutlet="selectionActions"></ng-container>
          } @else {
            <span class="text-[--cinema-text-dim]">Click and drag to draw selection</span>
          }
        }

        @case ('zoom') {
          <span class="text-[--cinema-text-dim]">Click to zoom in, Shift+Click to zoom out</span>
          <button class="px-2 py-0.5 rounded text-xs bg-[--cinema-panel] text-[--cinema-text-muted] border border-[--cinema-border] hover:text-[--cinema-primary] transition-colors"
            (click)="canvasService.zoomToFit()">
            Fit to View
          </button>
        }

        @case ('select') {
          <span class="text-[--cinema-text-dim]">Click to select objects, drag to move</span>
        }

        @case ('pan') {
          <span class="text-[--cinema-text-dim]">Click and drag to pan, or hold Space with any tool</span>
        }

        @default {
          <span class="text-[--cinema-text-dim]">Select a tool to begin</span>
        }
      }

      <!-- Active layer indicator (right side) -->
      <div class="ml-auto flex items-center gap-2">
        @if (canvasService.activeLayer(); as layer) {
          <span class="text-[--cinema-text-dim]">Layer:</span>
          <span class="text-[--cinema-text]">{{ layer.name }}</span>
          @if (layer.locked) {
            <span class="material-symbols-outlined text-[14px] text-[--cinema-error]">lock</span>
          }
        }
      </div>
    </div>

    <!-- Selection actions template -->
    <ng-template #selectionActions>
      <button class="px-2 py-0.5 rounded text-xs bg-[--cinema-panel] text-[--cinema-text-muted] border border-[--cinema-border] hover:text-[--cinema-primary] transition-colors"
        (click)="canvasService.selectionToMask()">
        <span class="material-symbols-outlined text-[12px] mr-0.5 align-middle">deblur</span> To Mask
      </button>
      <button class="px-2 py-0.5 rounded text-xs bg-[--cinema-panel] text-[--cinema-text-muted] border border-[--cinema-border] hover:text-[--cinema-error] transition-colors"
        (click)="canvasService.deleteWithinSelection()">
        <span class="material-symbols-outlined text-[12px] mr-0.5 align-middle">delete</span> Delete
      </button>
      <button class="px-2 py-0.5 rounded text-xs bg-[--cinema-panel] text-[--cinema-text-muted] border border-[--cinema-border] hover:text-[--cinema-primary] transition-colors"
        (click)="canvasService.clearSelection()">
        Clear
      </button>
    </ng-template>
  `
})
export class CanvasToolOptionsComponent {
  canvasService = inject(CanvasService);

  toolDisplayName(): string {
    const names: Record<string, string> = {
      'select': 'Select',
      'pan': 'Pan',
      'brush': 'Brush',
      'eraser': 'Eraser',
      'fill': 'Fill',
      'line': 'Line',
      'mask': 'Mask',
      'rect_select': 'Rect Select',
      'ellipse_select': 'Ellipse Select',
      'lasso_select': 'Lasso',
      'sam_select': 'SAM Auto',
      'zoom': 'Zoom',
    };
    return names[this.canvasService.activeTool()] ?? 'Tool';
  }
}
