
import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CanvasService, CanvasTool } from '../../services/canvas.service';

interface ToolDef {
  id: CanvasTool;
  icon: string;
  label: string;
  shortcut: string;
  divider?: boolean;
  disabled?: boolean;
  disabledTip?: string;
}

@Component({
  selector: 'app-canvas-toolbar',
  standalone: true,
  imports: [CommonModule],
  template: `
    <aside class="w-14 bg-[--cinema-bg] border-r border-[--cinema-border] flex flex-col items-center py-4 gap-1 z-40 shrink-0">
      @for (tool of tools; track tool.id) {
        @if (tool.divider) {
          <div class="h-px w-8 bg-[--cinema-border] mx-auto my-2"></div>
        }
        <button
          class="group relative flex items-center justify-center size-10 rounded-lg transition-all"
          [class]="canvasService.activeTool() === tool.id
            ? 'bg-[--cinema-primary]/15 text-[--cinema-primary] border border-[--cinema-primary]/30'
            : tool.disabled
              ? 'text-[--cinema-text-dim] border border-transparent cursor-not-allowed opacity-40'
              : 'text-[--cinema-text-muted] hover:text-[--cinema-primary] hover:bg-[--cinema-primary]/5 border border-transparent'"
          [disabled]="tool.disabled ?? false"
          (click)="selectTool(tool)">
          <span class="material-symbols-outlined text-[20px]">{{ tool.icon }}</span>
          <!-- Tooltip -->
          <span class="absolute left-14 bg-[--cinema-surface] text-[--cinema-text] text-xs px-2.5 py-1.5 rounded-lg border border-[--cinema-border] opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-50 font-display shadow-lg">
            {{ tool.disabled ? tool.disabledTip : tool.label }}
            @if (!tool.disabled) {
              <span class="text-[--cinema-text-dim] ml-1.5">{{ tool.shortcut }}</span>
            }
          </span>
        </button>
      }

      <!-- Spacer -->
      <div class="flex-1"></div>

      <!-- Undo/Redo -->
      <button
        class="flex items-center justify-center size-10 rounded-lg text-[--cinema-text-muted] hover:text-[--cinema-primary] hover:bg-[--cinema-primary]/5 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
        [disabled]="!canvasService.canUndo()"
        (click)="canvasService.undo()"
        title="Undo (Ctrl+Z)">
        <span class="material-symbols-outlined text-[20px]">undo</span>
      </button>
      <button
        class="flex items-center justify-center size-10 rounded-lg text-[--cinema-text-muted] hover:text-[--cinema-primary] hover:bg-[--cinema-primary]/5 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
        [disabled]="!canvasService.canRedo()"
        (click)="canvasService.redo()"
        title="Redo (Ctrl+Shift+Z)">
        <span class="material-symbols-outlined text-[20px]">redo</span>
      </button>
    </aside>
  `
})
export class CanvasToolbarComponent {
  canvasService = inject(CanvasService);

  tools: ToolDef[] = [
    { id: 'select', icon: 'arrow_selector_tool', label: 'Select', shortcut: 'V' },
    { id: 'pan', icon: 'pan_tool', label: 'Pan', shortcut: 'H' },
    { id: 'brush', icon: 'brush', label: 'Brush', shortcut: 'B', divider: true },
    { id: 'eraser', icon: 'ink_eraser', label: 'Eraser', shortcut: 'E' },
    { id: 'fill', icon: 'format_color_fill', label: 'Fill', shortcut: 'G' },
    { id: 'line', icon: 'pen_size_2', label: 'Line', shortcut: 'L' },
    { id: 'mask', icon: 'deblur', label: 'Mask Paint', shortcut: 'M', divider: true },
    { id: 'rect_select', icon: 'select', label: 'Rectangle Select', shortcut: 'R' },
    { id: 'ellipse_select', icon: 'radio_button_unchecked', label: 'Ellipse Select', shortcut: '' },
    { id: 'lasso_select', icon: 'lasso_select', label: 'Lasso Select', shortcut: '' },
    { id: 'sam_select', icon: 'auto_awesome', label: 'Coming Soon', shortcut: '', divider: true, disabled: true, disabledTip: 'SAM3 Auto-Mask — Coming Soon' },
    { id: 'zoom', icon: 'zoom_in', label: 'Zoom', shortcut: 'Z' },
  ];

  selectTool(tool: ToolDef) {
    if (tool.disabled) return;
    this.canvasService.setTool(tool.id);
  }
}
