
import { Injectable, signal, computed } from '@angular/core';
import * as fabric from 'fabric';

// ─── Tool Types ──────────────────────────────────────────────
export type CanvasTool =
  | 'select' | 'pan' | 'brush' | 'eraser' | 'fill' | 'line'
  | 'mask' | 'rect_select' | 'ellipse_select' | 'lasso_select'
  | 'sam_select' | 'zoom';

// Tools that use free drawing (PencilBrush)
const DRAWING_TOOLS: CanvasTool[] = ['brush', 'eraser', 'mask'];

// ─── Layer Model ─────────────────────────────────────────────
export interface CanvasLayer {
  id: string;
  name: string;
  type: 'image' | 'draw' | 'mask' | 'text';
  fabricGroup: fabric.Group;
  visible: boolean;
  locked: boolean;
  opacity: number;
  blendMode: GlobalCompositeOperation;
  zIndex: number;
}

// ─── Brush Options ───────────────────────────────────────────
export interface BrushOptions {
  size: number;
  color: string;
  opacity: number;
}

// ─── Selection Region ────────────────────────────────────────
export interface SelectionRegion {
  type: 'rect' | 'ellipse' | 'lasso';
  object: fabric.FabricObject;
}

// ─── Brush Cursor (rendered by viewport overlay) ─────────────
export interface BrushCursorState {
  x: number;
  y: number;
  size: number;
  visible: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class CanvasService {
  private _canvas: fabric.Canvas | null = null;

  // ─── Signals ─────────────────────────────────────────────
  readonly activeTool = signal<CanvasTool>('select');
  readonly activeLayerId = signal<string | null>(null);
  readonly layers = signal<CanvasLayer[]>([]);
  readonly zoom = signal(1);
  readonly brushOptions = signal<BrushOptions>({ size: 8, color: '#ffffff', opacity: 1 });
  readonly maskColor = signal<'white' | 'black'>('white');

  readonly hasSelection = signal(false);
  readonly selectionRegion = signal<SelectionRegion | null>(null);

  private undoStack: string[] = [];
  private redoStack: string[] = [];
  private maxHistory = 30;
  private savingState = false;
  readonly canUndo = signal(false);
  readonly canRedo = signal(false);

  private lineStart: fabric.Point | null = null;
  private linePreview: fabric.Line | null = null;
  private selStart: fabric.Point | null = null;
  private selPreview: fabric.FabricObject | null = null;
  private lassoPoints: fabric.Point[] = [];
  private isPanning = false;
  private panLastPos = { x: 0, y: 0 };
  private spaceHeld = false;

  readonly fillTolerance = signal(32);

  // Brush cursor state — rendered as overlay div by CanvasViewportComponent
  readonly brushCursor = signal<BrushCursorState>({ x: 0, y: 0, size: 8, visible: false });

  // ─── Frame / Crop Overlay ─────────────────────────────────
  readonly frameEnabled = signal(true);
  readonly frameAspectRatio = signal<string>('1:1');  // '1:1', '4:3', '3:4', '16:9', '9:16', 'free'
  readonly frameRect = signal({ x: 0, y: 0, w: 1024, h: 1024 });

  readonly FRAME_RATIOS: { label: string; value: string; ratio: number | null }[] = [
    { label: '1:1', value: '1:1', ratio: 1 },
    { label: '4:3', value: '4:3', ratio: 4 / 3 },
    { label: '3:4', value: '3:4', ratio: 3 / 4 },
    { label: '16:9', value: '16:9', ratio: 16 / 9 },
    { label: '9:16', value: '9:16', ratio: 9 / 16 },
    { label: '3:2', value: '3:2', ratio: 3 / 2 },
    { label: '2:3', value: '2:3', ratio: 2 / 3 },
    { label: 'Free', value: 'free', ratio: null },
  ];

  readonly activeLayer = computed(() => {
    const id = this.activeLayerId();
    return this.layers().find(l => l.id === id) ?? null;
  });

  get canvas(): fabric.Canvas | null {
    return this._canvas;
  }

  // ═══════════════════════════════════════════════════════════
  //  INIT / DESTROY
  // ═══════════════════════════════════════════════════════════
  initCanvas(canvasEl: HTMLCanvasElement, width: number, height: number) {
    if (this._canvas) this._canvas.dispose();

    this._canvas = new fabric.Canvas(canvasEl, {
      width,
      height,
      backgroundColor: 'transparent',
      selection: false,              // NEVER allow rubber-band multi-select
      preserveObjectStacking: true,
      skipTargetFind: true,          // Start with no object interaction
    });

    this._setupEventListeners();
    this._applyToolMode();
    this._saveState();
  }

  dispose() {
    if (this._canvas) {
      this._canvas.dispose();
      this._canvas = null;
    }
    this.layers.set([]);
    this.activeLayerId.set(null);
    this.undoStack = [];
    this.redoStack = [];
    this.canUndo.set(false);
    this.canRedo.set(false);
    this.brushCursor.set({ x: 0, y: 0, size: 8, visible: false });
  }

  // ═══════════════════════════════════════════════════════════
  //  TOOL SWITCHING
  // ═══════════════════════════════════════════════════════════
  setTool(tool: CanvasTool) {
    this.activeTool.set(tool);

    // Mask tool: auto-create & select mask layer
    if (tool === 'mask') {
      let maskLayer = this.layers().find(l => l.type === 'mask');
      if (!maskLayer) {
        maskLayer = this.addLayer('Mask', 'mask');
      }
      this.activeLayerId.set(maskLayer.id);
    }

    this._applyToolMode();
  }

  private _applyToolMode() {
    const c = this._canvas;
    if (!c) return;

    // Reset all modes
    c.isDrawingMode = false;
    c.defaultCursor = 'default';
    c.hoverCursor = 'default';
    this._clearSelPreview();

    const tool = this.activeTool();

    // ── CRITICAL: Tool isolation ─────────────────────────────
    // skipTargetFind = true → Fabric.js won't detect objects under mouse
    // This prevents object drag/select when using any non-select tool.
    // Like Photoshop: only the Move/Select tool interacts with objects.
    c.skipTargetFind = tool !== 'select';
    c.selection = false; // Always off — no rubber-band multi-select

    // Deselect any active object when leaving select tool
    if (tool !== 'select') {
      c.discardActiveObject();
    }

    // Hide brush cursor for non-drawing tools
    if (!DRAWING_TOOLS.includes(tool)) {
      this.brushCursor.update(v => ({ ...v, visible: false }));
    }

    switch (tool) {
      case 'select':
        c.defaultCursor = 'default';
        c.hoverCursor = 'move';
        break;

      case 'pan':
        c.defaultCursor = 'grab';
        break;

      case 'brush': {
        c.isDrawingMode = true;
        c.freeDrawingCursor = 'none'; // We render our own brush cursor
        const brush = new fabric.PencilBrush(c);
        const opts = this.brushOptions();
        brush.color = this._colorWithOpacity(opts.color, opts.opacity);
        brush.width = opts.size;
        c.freeDrawingBrush = brush;
        break;
      }

      case 'mask': {
        c.isDrawingMode = true;
        c.freeDrawingCursor = 'none';
        const mBrush = new fabric.PencilBrush(c);
        const mOpts = this.brushOptions();
        mBrush.color = this.maskColor() === 'white' ? 'rgba(255,0,0,0.5)' : 'rgba(0,100,255,0.5)';
        mBrush.width = mOpts.size;
        c.freeDrawingBrush = mBrush;
        break;
      }

      case 'eraser': {
        c.isDrawingMode = true;
        c.freeDrawingCursor = 'none';
        const eraser = new fabric.PencilBrush(c);
        const eopts = this.brushOptions();
        // Preview: semi-transparent light stripe during drawing
        eraser.color = `rgba(200,200,200,${Math.max(0.15, eopts.opacity * 0.3)})`;
        eraser.width = eopts.size;
        c.freeDrawingBrush = eraser;
        break;
      }

      case 'fill':
      case 'line':
        c.defaultCursor = 'crosshair';
        break;

      case 'rect_select':
      case 'ellipse_select':
      case 'lasso_select':
        c.defaultCursor = 'crosshair';
        break;

      case 'zoom':
        c.defaultCursor = 'zoom-in';
        break;

      case 'sam_select':
        c.defaultCursor = 'crosshair';
        break;
    }

    this._syncLayerInteractivity();
  }

  // ─── Core: sync which layers/objects are interactive ──────
  private _syncLayerInteractivity() {
    const c = this._canvas;
    if (!c) return;

    const tool = this.activeTool();
    const activeId = this.activeLayerId();
    const isSelectTool = tool === 'select';

    for (const layer of this.layers()) {
      const g = layer.fabricGroup;
      const isActive = layer.id === activeId;
      const canInteract = isActive && isSelectTool && !layer.locked;

      g.set({
        selectable: false,
        evented: canInteract,
        interactive: canInteract,
        subTargetCheck: canInteract,
      } as any);

      for (const child of g.getObjects()) {
        child.set({
          selectable: canInteract,
          evented: canInteract,
          hasControls: canInteract,
          hasBorders: canInteract,
        });
      }
    }
    c.requestRenderAll();
  }

  setActiveLayer(layerId: string) {
    this.activeLayerId.set(layerId);
    this._syncLayerInteractivity();
  }

  // ═══════════════════════════════════════════════════════════
  //  LAYER OPERATIONS
  // ═══════════════════════════════════════════════════════════
  addLayer(name: string, type: CanvasLayer['type']): CanvasLayer {
    const c = this._canvas!;

    const group = new fabric.Group([], {
      selectable: false,
      evented: false,
      interactive: true,
      subTargetCheck: true,
    } as any);
    c.add(group);

    const current = this.layers();
    const zIndex = current.length;
    const layer: CanvasLayer = {
      id: crypto.randomUUID(),
      name: name || `Layer ${zIndex + 1}`,
      type,
      fabricGroup: group,
      visible: true,
      locked: false,
      opacity: 1,
      blendMode: 'source-over',
      zIndex,
    };

    this.layers.update(prev => [...prev, layer]);
    this.activeLayerId.set(layer.id);
    this._reorderFabricObjects();
    this._syncLayerInteractivity();
    return layer;
  }

  addImageLayer(name: string, imageUrl: string): Promise<CanvasLayer> {
    return new Promise((resolve) => {
      const layer = this.addLayer(name, 'image');
      fabric.FabricImage.fromURL(imageUrl, { crossOrigin: 'anonymous' }).then((img) => {
        const c = this._canvas!;

        if (this.frameEnabled()) {
          const frame = this.frameRect();
          const scaleX = frame.w / (img.width || 1);
          const scaleY = frame.h / (img.height || 1);
          const scale = Math.min(scaleX, scaleY);
          const imgW = (img.width || 0) * scale;
          const imgH = (img.height || 0) * scale;

          // Scale image at origin, add to group, then reposition group to frame
          img.set({
            scaleX: scale,
            scaleY: scale,
            selectable: true,
            evented: true,
            hasControls: true,
            hasBorders: true,
          });
          layer.fabricGroup.add(img);

          // After FitContentLayout, move group so image is centered in frame
          layer.fabricGroup.set({
            left: frame.x + (frame.w - imgW) / 2,
            top: frame.y + (frame.h - imgH) / 2,
          });
          layer.fabricGroup.setCoords();
        } else {
          const scaleX = c.width! / (img.width || 1);
          const scaleY = c.height! / (img.height || 1);
          const scale = Math.min(scaleX, scaleY);
          img.set({
            scaleX: scale,
            scaleY: scale,
            left: (c.width! - (img.width || 0) * scale) / 2,
            top: (c.height! - (img.height || 0) * scale) / 2,
            selectable: true,
            evented: true,
            hasControls: true,
            hasBorders: true,
          });
          layer.fabricGroup.add(img);
        }

        c.requestRenderAll();
        this._syncLayerInteractivity();
        this._saveState();
        resolve(layer);
      });
    });
  }

  removeLayer(layerId: string) {
    const c = this._canvas;
    if (!c) return;
    const layer = this.layers().find(l => l.id === layerId);
    if (!layer) return;

    c.remove(layer.fabricGroup);
    this.layers.update(prev => prev.filter(l => l.id !== layerId));

    if (this.activeLayerId() === layerId) {
      const remaining = this.layers();
      this.activeLayerId.set(remaining.length > 0 ? remaining[remaining.length - 1].id : null);
    }
    this._reorderFabricObjects();
    this._syncLayerInteractivity();
    this._saveState();
  }

  async duplicateLayer(layerId: string) {
    const layer = this.layers().find(l => l.id === layerId);
    if (!layer || !this._canvas) return;

    const cloned = await layer.fabricGroup.clone();
    cloned.set({ selectable: false, evented: false, interactive: true, subTargetCheck: true } as any);
    this._canvas.add(cloned);

    const newLayer: CanvasLayer = {
      id: crypto.randomUUID(),
      name: layer.name + ' copy',
      type: layer.type,
      fabricGroup: cloned,
      visible: layer.visible,
      locked: layer.locked,
      opacity: layer.opacity,
      blendMode: layer.blendMode,
      zIndex: this.layers().length,
    };

    this.layers.update(prev => [...prev, newLayer]);
    this.activeLayerId.set(newLayer.id);
    this._reorderFabricObjects();
    this._syncLayerInteractivity();
    this._saveState();
  }

  reorderLayers(fromIndex: number, toIndex: number) {
    this.layers.update(prev => {
      const arr = [...prev];
      const [moved] = arr.splice(fromIndex, 1);
      arr.splice(toIndex, 0, moved);
      return arr.map((l, i) => ({ ...l, zIndex: i }));
    });
    this._reorderFabricObjects();
    this._saveState();
  }

  setLayerVisibility(layerId: string, visible: boolean) {
    this._updateLayer(layerId, l => {
      l.visible = visible;
      l.fabricGroup.set({ visible });
    });
    this._canvas?.requestRenderAll();
  }

  setLayerLocked(layerId: string, locked: boolean) {
    this._updateLayer(layerId, l => { l.locked = locked; });
    this._syncLayerInteractivity();
  }

  setLayerOpacity(layerId: string, opacity: number) {
    this._updateLayer(layerId, l => {
      l.opacity = opacity;
      l.fabricGroup.set({ opacity });
    });
    this._canvas?.requestRenderAll();
  }

  setLayerBlendMode(layerId: string, blendMode: GlobalCompositeOperation) {
    this._updateLayer(layerId, l => {
      l.blendMode = blendMode;
      l.fabricGroup.set({ globalCompositeOperation: blendMode });
    });
    this._canvas?.requestRenderAll();
  }

  renameLayer(layerId: string, name: string) {
    this._updateLayer(layerId, l => { l.name = name; });
  }

  mergeDown(layerId: string) {
    const all = this.layers();
    const idx = all.findIndex(l => l.id === layerId);
    if (idx <= 0 || !this._canvas) return;

    const upper = all[idx];
    const lower = all[idx - 1];
    const objects = [...upper.fabricGroup.getObjects()];
    for (const obj of objects) {
      upper.fabricGroup.remove(obj);
      lower.fabricGroup.add(obj);
    }
    this._canvas.remove(upper.fabricGroup);
    this.layers.update(prev => prev.filter(l => l.id !== layerId));
    this.activeLayerId.set(lower.id);
    this._reorderFabricObjects();
    this._syncLayerInteractivity();
    this._saveState();
  }

  /** Remove all canvas objects not belonging to any layer group */
  removeOrphanedObjects() {
    const c = this._canvas;
    if (!c) return;
    const layerGroups = new Set(this.layers().map(l => l.fabricGroup));
    const orphans = c.getObjects().filter(obj => !layerGroups.has(obj as any));
    for (const obj of orphans) {
      c.remove(obj);
    }
    c.requestRenderAll();
  }

  /** Remove all layers and all canvas objects */
  clearAll() {
    const c = this._canvas;
    if (!c) return;
    // Remove all layer groups
    for (const layer of this.layers()) {
      c.remove(layer.fabricGroup);
    }
    this.layers.set([]);
    this.activeLayerId.set(null);
    // Remove any remaining orphaned objects
    const remaining = [...c.getObjects()];
    for (const obj of remaining) {
      c.remove(obj);
    }
    c.requestRenderAll();
    this._saveState();
  }

  flattenAllLayers(): CanvasLayer | null {
    const all = this.layers();
    if (all.length === 0 || !this._canvas) return null;

    const base = all[0];
    for (let i = 1; i < all.length; i++) {
      const layer = all[i];
      const objects = [...layer.fabricGroup.getObjects()];
      for (const obj of objects) {
        layer.fabricGroup.remove(obj);
        base.fabricGroup.add(obj);
      }
      this._canvas.remove(layer.fabricGroup);
    }
    this.layers.set([base]);
    this.activeLayerId.set(base.id);
    this._reorderFabricObjects();
    this._syncLayerInteractivity();
    this._saveState();
    return base;
  }

  private _updateLayer(layerId: string, fn: (layer: CanvasLayer) => void) {
    this.layers.update(prev => prev.map(l => {
      if (l.id === layerId) {
        fn(l);
        return { ...l };
      }
      return l;
    }));
  }

  private _reorderFabricObjects() {
    const c = this._canvas;
    if (!c) return;
    const all = this.layers();
    for (const layer of all) {
      c.remove(layer.fabricGroup);
    }
    for (const layer of all) {
      c.add(layer.fabricGroup);
    }
    c.requestRenderAll();
  }

  // ═══════════════════════════════════════════════════════════
  //  ZOOM & PAN
  // ═══════════════════════════════════════════════════════════
  setZoom(level: number) {
    const c = this._canvas;
    if (!c) return;
    const clamped = Math.max(0.1, Math.min(10, level));
    const center = c.getCenterPoint();
    c.zoomToPoint(center, clamped);
    this.zoom.set(clamped);
  }

  zoomToFit() {
    this.setZoom(1);
    this._canvas?.setViewportTransform([1, 0, 0, 1, 0, 0]);
  }

  handleWheel(e: WheelEvent) {
    const c = this._canvas;
    if (!c) return;
    e.preventDefault();

    let z = this.zoom();
    z *= 0.999 ** e.deltaY;
    z = Math.max(0.1, Math.min(10, z));

    c.zoomToPoint(new fabric.Point(e.offsetX, e.offsetY), z);
    this.zoom.set(z);
  }

  startPan(e: MouseEvent) {
    this.isPanning = true;
    this.panLastPos = { x: e.clientX, y: e.clientY };
    if (this._canvas) this._canvas.defaultCursor = 'grabbing';
  }

  doPan(e: MouseEvent) {
    if (!this.isPanning || !this._canvas) return;
    const vpt = this._canvas.viewportTransform!;
    vpt[4] += e.clientX - this.panLastPos.x;
    vpt[5] += e.clientY - this.panLastPos.y;
    this.panLastPos = { x: e.clientX, y: e.clientY };
    this._canvas.requestRenderAll();
  }

  endPan() {
    this.isPanning = false;
    this._applyToolMode();
  }

  setSpaceHeld(held: boolean) {
    this.spaceHeld = held;
    if (held && this.activeTool() !== 'pan') {
      if (this._canvas) {
        this._canvas.defaultCursor = 'grab';
        // Temporarily hide brush cursor while panning
        this.brushCursor.update(v => ({ ...v, visible: false }));
      }
    } else if (!held) {
      this._applyToolMode();
    }
  }

  get isSpaceHeld() { return this.spaceHeld; }

  // ═══════════════════════════════════════════════════════════
  //  BRUSH OPTIONS
  // ═══════════════════════════════════════════════════════════
  setBrushSize(size: number) {
    this.brushOptions.update(o => ({ ...o, size }));
    if (this._canvas?.freeDrawingBrush) {
      this._canvas.freeDrawingBrush.width = size;
    }
    // Update cursor size immediately
    this.brushCursor.update(v => ({ ...v, size: size * this.zoom() }));
  }

  setBrushColor(color: string) {
    this.brushOptions.update(o => ({ ...o, color }));
    if (this._canvas?.freeDrawingBrush && this.activeTool() === 'brush') {
      this._canvas.freeDrawingBrush.color = this._colorWithOpacity(color, this.brushOptions().opacity);
    }
  }

  setBrushOpacity(opacity: number) {
    this.brushOptions.update(o => ({ ...o, opacity }));
    const tool = this.activeTool();
    if (this._canvas?.freeDrawingBrush) {
      if (tool === 'brush') {
        this._canvas.freeDrawingBrush.color = this._colorWithOpacity(this.brushOptions().color, opacity);
      } else if (tool === 'eraser') {
        this._canvas.freeDrawingBrush.color = `rgba(200,200,200,${Math.max(0.15, opacity * 0.3)})`;
      }
    }
  }

  setMaskColor(color: 'white' | 'black') {
    this.maskColor.set(color);
    if (this._canvas?.freeDrawingBrush && this.activeTool() === 'mask') {
      this._canvas.freeDrawingBrush.color = color === 'white' ? 'rgba(255,0,0,0.5)' : 'rgba(0,100,255,0.5)';
    }
  }

  // Convert hex color + opacity to rgba string
  private _colorWithOpacity(hex: string, opacity: number): string {
    const rgb = this._hexToRgba(hex);
    return `rgba(${rgb.r},${rgb.g},${rgb.b},${opacity})`;
  }

  // ═══════════════════════════════════════════════════════════
  //  UNDO / REDO
  // ═══════════════════════════════════════════════════════════
  private _saveState() {
    if (this.savingState || !this._canvas) return;
    this.savingState = true;

    const json = JSON.stringify(this._canvas.toJSON());
    this.undoStack.push(json);
    if (this.undoStack.length > this.maxHistory) this.undoStack.shift();
    this.redoStack = [];
    this.canUndo.set(this.undoStack.length > 1);
    this.canRedo.set(false);

    this.savingState = false;
  }

  undo() {
    if (this.undoStack.length <= 1 || !this._canvas) return;
    const current = this.undoStack.pop()!;
    this.redoStack.push(current);
    const prev = this.undoStack[this.undoStack.length - 1];
    this.savingState = true;
    this._canvas.loadFromJSON(prev).then(() => {
      this._canvas!.requestRenderAll();
      this.savingState = false;
      this.canUndo.set(this.undoStack.length > 1);
      this.canRedo.set(this.redoStack.length > 0);
    });
  }

  redo() {
    if (this.redoStack.length === 0 || !this._canvas) return;
    const next = this.redoStack.pop()!;
    this.undoStack.push(next);
    this.savingState = true;
    this._canvas.loadFromJSON(next).then(() => {
      this._canvas!.requestRenderAll();
      this.savingState = false;
      this.canUndo.set(this.undoStack.length > 1);
      this.canRedo.set(this.redoStack.length > 0);
    });
  }

  // ═══════════════════════════════════════════════════════════
  //  EXPORT
  //  All exports hide overlays (selection, mask) so only
  //  actual image/draw content is captured.
  // ═══════════════════════════════════════════════════════════

  /** Temporarily hide non-content overlays, returns restore function */
  private _hideOverlaysForExport(): () => void {
    const hidden: { obj: fabric.FabricObject; wasVisible: boolean }[] = [];

    // Hide selection overlay (purple rect/ellipse/lasso)
    const sel = this.selectionRegion();
    if (sel?.object) {
      hidden.push({ obj: sel.object, wasVisible: sel.object.visible !== false });
      sel.object.set({ visible: false });
    }

    // Hide mask layers (red/blue visualization — not image content)
    for (const l of this.layers()) {
      if (l.type === 'mask' && l.fabricGroup.visible !== false) {
        hidden.push({ obj: l.fabricGroup, wasVisible: true });
        l.fabricGroup.set({ visible: false });
      }
    }

    this._canvas?.requestRenderAll();

    return () => {
      for (const { obj, wasVisible } of hidden) {
        obj.set({ visible: wasVisible });
      }
      this._canvas?.requestRenderAll();
    };
  }

  exportCanvasAsBase64(format: 'png' | 'jpeg' = 'png'): string {
    if (!this._canvas) return '';
    const restore = this._hideOverlaysForExport();
    const result = this._canvas.toDataURL({ format, multiplier: 1 });
    restore();
    return result;
  }

  /**
   * Export mask as black/white image (frame-based).
   * White = regenerate area, Black = keep area.
   * Sources: mask layer strokes (red→white, blue→black) + selection region (→white).
   */
  exportMaskAsBase64(): string {
    const c = this._canvas;
    if (!c) return '';

    const maskLayers = this.layers().filter(l => l.type === 'mask');
    const sel = this.selectionRegion();
    if (maskLayers.length === 0 && !sel) return '';

    // --- Save state ---
    const origBg = c.backgroundColor;
    const layerVisibility = new Map<string, boolean>();
    for (const l of this.layers()) {
      layerVisibility.set(l.id, l.fabricGroup.visible !== false);
    }

    // --- Configure canvas for mask rendering ---
    c.backgroundColor = '#000000';

    // Hide ALL layers (we only want mask content)
    for (const l of this.layers()) {
      l.fabricGroup.set({ visible: l.type === 'mask' });
    }

    // Recolor mask strokes: red (edit) → white, blue (keep) → black
    const strokeBackups: { obj: fabric.FabricObject; origStroke: any }[] = [];
    for (const ml of maskLayers) {
      for (const obj of ml.fabricGroup.getObjects()) {
        const origStroke = (obj as any).stroke;
        strokeBackups.push({ obj, origStroke });
        const isKeep = typeof origStroke === 'string' && origStroke.includes('0,0,255');
        obj.set({ stroke: isKeep ? '#000000' : '#ffffff' } as any);
      }
    }

    // Hide original selection overlay (purple)
    const selWasVisible = sel?.object?.visible !== false;
    if (sel?.object) sel.object.set({ visible: false });

    // Add white-filled clone of selection shape as mask area
    let selClone: fabric.FabricObject | null = null;
    if (sel) {
      const obj = sel.object;
      const fillOpts = { fill: '#ffffff', stroke: 'transparent', strokeWidth: 0, selectable: false, evented: false };

      if (sel.type === 'rect') {
        selClone = new fabric.Rect({
          left: obj.left, top: obj.top,
          width: ((obj as any).width || 0) * (obj.scaleX || 1),
          height: ((obj as any).height || 0) * (obj.scaleY || 1),
          ...fillOpts,
        });
      } else if (sel.type === 'ellipse') {
        selClone = new fabric.Ellipse({
          left: obj.left, top: obj.top,
          rx: ((obj as any).rx || 0) * (obj.scaleX || 1),
          ry: ((obj as any).ry || 0) * (obj.scaleY || 1),
          ...fillOpts,
        });
      } else if (sel.type === 'lasso' && obj instanceof fabric.Path) {
        const pathData = (obj as any).path;
        if (pathData) {
          selClone = new fabric.Path(fabric.util.joinPath(pathData), fillOpts);
        }
      }
      if (selClone) c.add(selClone);
    }

    c.requestRenderAll();

    // --- Export frame area ---
    const frame = this.frameRect();
    const result = c.toDataURL({
      format: 'png' as 'png',
      left: frame.x,
      top: frame.y,
      width: frame.w,
      height: frame.h,
      multiplier: 1,
    });

    // --- Restore everything ---
    if (selClone) c.remove(selClone);
    if (sel?.object) sel.object.set({ visible: selWasVisible });

    for (const { obj, origStroke } of strokeBackups) {
      obj.set({ stroke: origStroke } as any);
    }

    c.backgroundColor = origBg;
    for (const l of this.layers()) {
      l.fabricGroup.set({ visible: layerVisibility.get(l.id) ?? true });
    }

    c.requestRenderAll();
    return result;
  }

  exportActiveLayerAsBase64(): string {
    const layer = this.activeLayer();
    if (!layer || !this._canvas) return '';

    const restore = this._hideOverlaysForExport();
    const allLayers = this.layers();
    const otherLayers = allLayers.filter(l => l.id !== layer.id && l.type !== 'mask');
    otherLayers.forEach(l => l.fabricGroup.set({ visible: false }));

    const data = this._canvas.toDataURL({ format: 'png', multiplier: 1 });

    otherLayers.forEach(l => l.fabricGroup.set({ visible: l.visible }));
    restore();
    return data;
  }

  // ═══════════════════════════════════════════════════════════
  //  EVENT LISTENERS
  // ═══════════════════════════════════════════════════════════
  private _setupEventListeners() {
    const c = this._canvas!;

    // ── Path created (brush / eraser / mask strokes) ────────
    c.on('path:created', (e: any) => {
      const path = e.path as fabric.Path;
      const tool = this.activeTool();
      const layer = this.activeLayer();

      if (!layer || layer.locked) {
        c.remove(path);
        return;
      }

      if (tool === 'eraser') {
        const opacity = this.brushOptions().opacity;
        path.set({
          globalCompositeOperation: 'destination-out',
          stroke: `rgba(0,0,0,${opacity})`,
          selectable: false,
          evented: false,
        });
      } else {
        path.set({ selectable: false, evented: false });
      }

      // Move path from canvas root into the active layer group
      c.remove(path);
      layer.fabricGroup.add(path);
      c.requestRenderAll();
      this._saveState();
    });

    // ── Mouse down ──────────────────────────────────────────
    c.on('mouse:down', (opt) => {
      const e = opt.e as MouseEvent;
      const tool = this.activeTool();

      // Space-drag pan override (works from any tool)
      if (this.spaceHeld || tool === 'pan') {
        this.startPan(e);
        return;
      }

      // Select tool: let Fabric.js handle it (skipTargetFind=false)
      if (tool === 'select') return;

      // Drawing tools: Fabric handles via isDrawingMode
      if (DRAWING_TOOLS.includes(tool)) {
        // Alt+click toggles mask mode
        if (tool === 'mask' && e.altKey) {
          this.setMaskColor(this.maskColor() === 'white' ? 'black' : 'white');
        }
        return;
      }

      const pointer = c.getScenePoint(e);

      if (tool === 'line') { this.lineStart = pointer; return; }
      if (tool === 'fill') { this._doFloodFill(pointer); return; }

      if (tool === 'zoom') {
        const newZoom = e.shiftKey ? this.zoom() / 1.3 : this.zoom() * 1.3;
        const pt = new fabric.Point(e.offsetX, e.offsetY);
        const clamped = Math.max(0.1, Math.min(10, newZoom));
        c.zoomToPoint(pt, clamped);
        this.zoom.set(clamped);
        return;
      }

      if (tool === 'rect_select' || tool === 'ellipse_select') {
        // Clear previous selection when starting a new one
        this.clearSelection();
        this.selStart = pointer;
        return;
      }
      if (tool === 'lasso_select') {
        this.clearSelection();
        this.lassoPoints = [pointer];
        return;
      }
    });

    // ── Mouse move ──────────────────────────────────────────
    c.on('mouse:move', (opt) => {
      const e = opt.e as MouseEvent;
      const tool = this.activeTool();

      // Always update brush cursor position for drawing tools
      if (DRAWING_TOOLS.includes(tool) && !this.spaceHeld && !this.isPanning) {
        this.brushCursor.set({
          x: e.offsetX,
          y: e.offsetY,
          size: this.brushOptions().size * this.zoom(),
          visible: true,
        });
      }

      if (this.isPanning || (this.spaceHeld && e.buttons === 1)) {
        this.doPan(e);
        return;
      }

      const pointer = c.getScenePoint(e);

      // Line preview
      if (tool === 'line' && this.lineStart) {
        if (this.linePreview) c.remove(this.linePreview);
        this.linePreview = new fabric.Line(
          [this.lineStart.x, this.lineStart.y, pointer.x, pointer.y],
          { stroke: this.brushOptions().color, strokeWidth: this.brushOptions().size, selectable: false, evented: false }
        );
        c.add(this.linePreview);
        c.requestRenderAll();
        return;
      }

      // Rect / Ellipse selection preview
      if ((tool === 'rect_select' || tool === 'ellipse_select') && this.selStart && e.buttons === 1) {
        if (this.selPreview) c.remove(this.selPreview);
        const left = Math.min(this.selStart.x, pointer.x);
        const top = Math.min(this.selStart.y, pointer.y);
        const w = Math.abs(pointer.x - this.selStart.x);
        const h = Math.abs(pointer.y - this.selStart.y);

        const opts = {
          fill: 'rgba(122,31,249,0.15)',
          stroke: '#7a1ff9', strokeWidth: 1,
          strokeDashArray: [6, 4],
          selectable: false, evented: false,
        };

        this.selPreview = tool === 'rect_select'
          ? new fabric.Rect({ left, top, width: w, height: h, ...opts })
          : new fabric.Ellipse({ left, top, rx: w / 2, ry: h / 2, ...opts });
        c.add(this.selPreview);
        c.requestRenderAll();
        return;
      }

      // Lasso preview
      if (tool === 'lasso_select' && this.lassoPoints.length > 0 && e.buttons === 1) {
        this.lassoPoints.push(pointer);
        if (this.selPreview) c.remove(this.selPreview);
        const pathStr = this.lassoPoints.map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`)).join(' ');
        this.selPreview = new fabric.Path(pathStr, {
          fill: 'rgba(122,31,249,0.15)', stroke: '#7a1ff9', strokeWidth: 1,
          strokeDashArray: [6, 4], selectable: false, evented: false,
        });
        c.add(this.selPreview);
        c.requestRenderAll();
        return;
      }
    });

    // ── Mouse up ────────────────────────────────────────────
    c.on('mouse:up', (opt) => {
      const e = opt.e as MouseEvent;
      const tool = this.activeTool();

      if (this.isPanning) { this.endPan(); return; }

      const pointer = c.getScenePoint(e);

      // Line finalize
      if (tool === 'line' && this.lineStart) {
        if (this.linePreview) c.remove(this.linePreview);
        const line = new fabric.Line(
          [this.lineStart.x, this.lineStart.y, pointer.x, pointer.y],
          { stroke: this.brushOptions().color, strokeWidth: this.brushOptions().size, selectable: false, evented: false }
        );
        const layer = this.activeLayer();
        if (layer && !layer.locked) {
          layer.fabricGroup.add(line);
        } else {
          c.add(line);
        }
        this.lineStart = null;
        this.linePreview = null;
        c.requestRenderAll();
        this._saveState();
        return;
      }

      // Rect/Ellipse selection finalize
      if ((tool === 'rect_select' || tool === 'ellipse_select') && this.selStart) {
        if (this.selPreview) {
          this.selectionRegion.set({ type: tool === 'rect_select' ? 'rect' : 'ellipse', object: this.selPreview });
          this.hasSelection.set(true);
        }
        this.selStart = null;
        this.selPreview = null;
        return;
      }

      // Lasso finalize
      if (tool === 'lasso_select' && this.lassoPoints.length > 2) {
        if (this.selPreview) c.remove(this.selPreview);
        const pathStr = this.lassoPoints.map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`)).join(' ') + ' Z';
        const lasso = new fabric.Path(pathStr, {
          fill: 'rgba(122,31,249,0.15)', stroke: '#7a1ff9', strokeWidth: 1,
          strokeDashArray: [6, 4], selectable: false, evented: false,
        });
        c.add(lasso);
        this.selectionRegion.set({ type: 'lasso', object: lasso });
        this.hasSelection.set(true);
        this.lassoPoints = [];
        this.selPreview = null;
        c.requestRenderAll();
        return;
      }
    });

    // ── Object modified (drag/scale/rotate via select tool) ──
    c.on('object:modified', () => {
      this._saveState();
    });

    // ── Mouse leaves canvas — hide brush cursor ─────────────
    c.on('mouse:out', () => {
      this.brushCursor.update(v => ({ ...v, visible: false }));
    });
  }

  // ═══════════════════════════════════════════════════════════
  //  SELECTION ACTIONS
  // ═══════════════════════════════════════════════════════════
  clearSelection() {
    const sel = this.selectionRegion();
    if (sel && this._canvas) {
      this._canvas.remove(sel.object);
      this._canvas.requestRenderAll();
    }
    this.selectionRegion.set(null);
    this.hasSelection.set(false);
  }

  async selectionToMask() {
    const sel = this.selectionRegion();
    if (!sel) return;

    let maskLayer = this.layers().find(l => l.type === 'mask');
    if (!maskLayer) {
      maskLayer = this.addLayer('Mask', 'mask');
    }

    const clone = await sel.object.clone();
    clone.set({ fill: 'rgba(255,0,0,0.5)', stroke: 'transparent', selectable: false, evented: false });
    maskLayer.fabricGroup.add(clone);

    this.clearSelection();
    this._canvas?.requestRenderAll();
    this._saveState();
  }

  deleteWithinSelection() {
    this.clearSelection();
    this._saveState();
  }

  invertSelection() {
    // TODO
  }

  private _clearSelPreview() {
    if (this.selPreview && this._canvas) {
      this._canvas.remove(this.selPreview);
      this.selPreview = null;
    }
  }

  // ═══════════════════════════════════════════════════════════
  //  FLOOD FILL
  // ═══════════════════════════════════════════════════════════
  private _doFloodFill(point: fabric.Point) {
    const c = this._canvas;
    if (!c) return;
    const layer = this.activeLayer();
    if (!layer || layer.locked) return;

    const ctx = c.getContext() as CanvasRenderingContext2D;
    const w = c.width!;
    const h = c.height!;
    const imageData = ctx.getImageData(0, 0, w, h);
    const data = imageData.data;

    const x = Math.round(point.x);
    const y = Math.round(point.y);
    if (x < 0 || x >= w || y < 0 || y >= h) return;

    const targetIdx = (y * w + x) * 4;
    const tR = data[targetIdx], tG = data[targetIdx + 1], tB = data[targetIdx + 2], tA = data[targetIdx + 3];

    const fill = this._hexToRgba(this.brushOptions().color);
    const tol = this.fillTolerance();

    if (Math.abs(tR - fill.r) < 5 && Math.abs(tG - fill.g) < 5 && Math.abs(tB - fill.b) < 5) return;

    const visited = new Uint8Array(w * h);
    const stack: number[] = [x, y];

    while (stack.length > 0) {
      const cy = stack.pop()!;
      const cx = stack.pop()!;
      if (cx < 0 || cx >= w || cy < 0 || cy >= h) continue;
      const idx = cy * w + cx;
      if (visited[idx]) continue;
      visited[idx] = 1;

      const pi = idx * 4;
      if (Math.abs(data[pi] - tR) <= tol && Math.abs(data[pi+1] - tG) <= tol &&
          Math.abs(data[pi+2] - tB) <= tol && Math.abs(data[pi+3] - tA) <= tol) {
        data[pi] = fill.r; data[pi+1] = fill.g; data[pi+2] = fill.b; data[pi+3] = 255;
        stack.push(cx+1, cy, cx-1, cy, cx, cy+1, cx, cy-1);
      }
    }

    ctx.putImageData(imageData, 0, 0);
    this._saveState();
  }

  private _hexToRgba(hex: string): { r: number; g: number; b: number; a: number } {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? { r: parseInt(result[1], 16), g: parseInt(result[2], 16), b: parseInt(result[3], 16), a: 255 }
      : { r: 255, g: 255, b: 255, a: 255 };
  }

  // ═══════════════════════════════════════════════════════════
  //  DROP IMAGE / PERSISTENCE
  // ═══════════════════════════════════════════════════════════
  handleDroppedFile(file: File) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const url = e.target?.result as string;
      this.addImageLayer(file.name, url);
    };
    reader.readAsDataURL(file);
  }

  saveDraft(sceneId: string) {
    if (!this._canvas) return;
    try {
      const json = JSON.stringify(this._canvas.toJSON());
      localStorage.setItem(`vellumium_canvas_draft_${sceneId}`, json);
    } catch { /* quota exceeded */ }
  }

  async loadDraft(sceneId: string): Promise<boolean> {
    if (!this._canvas) return false;
    const json = localStorage.getItem(`vellumium_canvas_draft_${sceneId}`);
    if (!json) return false;
    try {
      await this._canvas.loadFromJSON(json);
      this._canvas.requestRenderAll();
      return true;
    } catch {
      return false;
    }
  }

  clearDraft(sceneId: string) {
    localStorage.removeItem(`vellumium_canvas_draft_${sceneId}`);
  }

  // ═══════════════════════════════════════════════════════════
  //  FRAME / CROP OVERLAY
  //  All frame coordinates are in fabric.js WORLD space.
  //  Convert to screen space only for rendering overlays.
  // ═══════════════════════════════════════════════════════════
  setFrameAspectRatio(value: string) {
    this.frameAspectRatio.set(value);
    const preset = this.FRAME_RATIOS.find(r => r.value === value);
    if (preset?.ratio) {
      const currentFrame = this.frameRect();
      const newH = Math.round(currentFrame.w / preset.ratio);
      this.frameRect.set({ ...currentFrame, h: newH });
    }
    this.centerFrame();
  }

  setFrameSize(w: number, h: number) {
    const frame = this.frameRect();
    this.frameRect.set({ ...frame, w, h });
    this.centerFrame();
  }

  toggleFrame() {
    this.frameEnabled.update(v => !v);
  }

  // Center frame in viewport (sets world-space x,y so frame appears centered on screen)
  centerFrame() {
    if (!this._canvas) return;
    const vpt = this._canvas.viewportTransform || [1, 0, 0, 1, 0, 0];
    const zoom = vpt[0];
    // Canvas element size in screen pixels
    const cw = this._canvas.getWidth();
    const ch = this._canvas.getHeight();
    const frame = this.frameRect();
    // Center of viewport in world space
    const centerWorldX = (cw / 2 - vpt[4]) / zoom;
    const centerWorldY = (ch / 2 - vpt[5]) / zoom;
    this.frameRect.set({
      ...frame,
      x: centerWorldX - frame.w / 2,
      y: centerWorldY - frame.h / 2,
    });
  }

  // Convert world-space frame to screen pixels for overlay rendering
  getFrameScreenRect(): { fx: number; fy: number; fw: number; fh: number } {
    const frame = this.frameRect();
    const vpt = this._canvas?.viewportTransform || [1, 0, 0, 1, 0, 0];
    const zoom = vpt[0];
    return {
      fx: frame.x * zoom + vpt[4],
      fy: frame.y * zoom + vpt[5],
      fw: frame.w * zoom,
      fh: frame.h * zoom,
    };
  }

  // Export only the frame area — hides overlays (selection, mask) for clean capture
  exportFrameAsBase64(fmt: 'png' | 'jpeg' = 'png'): string {
    if (!this._canvas) return '';
    const restore = this._hideOverlaysForExport();
    const frame = this.frameRect();
    const result = this._canvas.toDataURL({
      format: fmt,
      left: frame.x,
      top: frame.y,
      width: frame.w,
      height: frame.h,
      multiplier: 1,
    });
    restore();
    return result;
  }
}
