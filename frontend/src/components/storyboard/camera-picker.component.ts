
import {
  Component, ElementRef, ViewChild, AfterViewInit, OnDestroy,
  inject, signal, computed,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { StylePresetService } from '../../services/style-preset.service';
import { CAMERA_PRESETS, CAMERA_DISTANCES, CameraPreset, CameraDistancePreset } from '../../models/style-presets';

const HEMISPHERE_RADIUS = 2.0;
const MARKER_RADIUS = 0.09;

// Theme colors
const COL = {
  default:         new THREE.Color(0x555577),
  defaultEmissive: new THREE.Color(0x111122),
  hover:           new THREE.Color(0x9292c9),
  hoverEmissive:   new THREE.Color(0x333355),
  selected:        new THREE.Color(0x2b2bee),
  selectedEmissive:new THREE.Color(0x1515aa),
  wire:            new THREE.Color(0x232348),
  primary:         new THREE.Color(0x2b2bee),
  subject:         new THREE.Color(0x2b2bee),
  ground:          new THREE.Color(0x191933),
};

@Component({
  selector: 'app-camera-picker',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="space-y-2 p-3 bg-[--cinema-bg] rounded-lg border border-[--cinema-border]">
      <!-- Header -->
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-1.5">
          <span class="material-symbols-outlined text-[14px] text-[--cinema-primary]">videocam</span>
          <span class="text-[10px] font-display font-semibold text-[--cinema-primary] uppercase tracking-wider">Camera</span>
        </div>
        <button (click)="resetCamera()"
          class="text-[9px] text-[--cinema-text-dim] hover:text-[--cinema-primary] transition-colors">Reset</button>
      </div>

      <!-- 3D Hemisphere -->
      <div class="relative rounded-lg overflow-hidden border border-[--cinema-border]/50">
        <canvas #threeCanvas class="block w-full"></canvas>
        @if (!presetService.selectedCameraPreset()) {
          <div class="absolute bottom-2 left-1/2 -translate-x-1/2 text-[8px] text-[--cinema-text-dim]/40 pointer-events-none whitespace-nowrap">
            Click a point to set camera angle
          </div>
        }
      </div>

      <!-- Selected Label -->
      <div class="text-center h-4">
        @if (selectedLabel(); as label) {
          <span class="text-[10px] text-[--cinema-primary]/70 italic">{{ label }}</span>
        }
      </div>

      <!-- Distance -->
      <div class="flex items-center gap-2">
        <span class="text-[9px] text-[--cinema-text-muted] uppercase tracking-wider shrink-0 w-14">Distance</span>
        <div class="flex flex-1 gap-1">
          @for (d of distances; track d.id) {
            <button
              class="flex-1 py-1.5 text-[10px] rounded border transition-all flex items-center justify-center gap-1"
              [class]="presetService.selectedCameraDistance()?.id === d.id
                ? 'bg-[--cinema-primary]/15 text-[--cinema-primary] border-[--cinema-primary]/30'
                : 'text-[--cinema-text-muted] border-[--cinema-border] hover:border-[--cinema-primary]/20'"
              (click)="selectDistance(d)">
              <span class="material-symbols-outlined text-[12px]">{{ d.icon }}</span>
              {{ d.label }}
            </button>
          }
        </div>
      </div>
    </div>
  `
})
export class CameraPickerComponent implements AfterViewInit, OnDestroy {
  @ViewChild('threeCanvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;

  presetService = inject(StylePresetService);
  distances = CAMERA_DISTANCES;

  selectedLabel = computed(() => {
    const cam = this.presetService.selectedCameraPreset();
    const dist = this.presetService.selectedCameraDistance();
    const parts: string[] = [];
    if (cam) parts.push(cam.label);
    if (dist) parts.push(dist.label);
    return parts.length > 0 ? parts.join(' · ') : '';
  });

  // ── Three.js internals ──
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private raycaster = new THREE.Raycaster();
  private mouse = new THREE.Vector2();
  private markerMeshes: THREE.Mesh[] = [];
  private glowMeshes = new Map<string, THREE.Mesh>();
  private viewRay: THREE.Line | null = null;
  private animId = 0;
  private hoveredMesh: THREE.Mesh | null = null;
  private pointerDownPos = { x: 0, y: 0 };
  private wasDragging = false;

  ngAfterViewInit() {
    requestAnimationFrame(() => {
      this.initScene();
      this.animate();
      this.bindEvents();
    });
  }

  ngOnDestroy() {
    cancelAnimationFrame(this.animId);
    const c = this.canvasRef.nativeElement;
    c.removeEventListener('pointerdown', this.onPointerDown);
    c.removeEventListener('pointermove', this.onPointerMove);
    c.removeEventListener('pointerup', this.onPointerUp);
    this.controls?.dispose();
    this.renderer?.dispose();
    this.scene?.traverse(obj => {
      if (obj instanceof THREE.Mesh) {
        obj.geometry.dispose();
        if (obj.material instanceof THREE.Material) obj.material.dispose();
      }
      if (obj instanceof THREE.Line) {
        obj.geometry.dispose();
        if (obj.material instanceof THREE.Material) obj.material.dispose();
      }
    });
  }

  resetCamera() {
    this.presetService.resetCamera();
    this.markerMeshes.forEach(m => this.setMarkerVisual(m, 'default'));
    this.glowMeshes.forEach(g => g.visible = false);
    this.updateViewRay(null);
  }

  selectDistance(d: CameraDistancePreset) {
    const current = this.presetService.selectedCameraDistance();
    this.presetService.selectCameraDistance(current?.id === d.id ? null : d);
  }

  // ── Scene Setup ──

  private initScene() {
    const canvas = this.canvasRef.nativeElement;
    const parent = canvas.parentElement!;
    const w = parent.clientWidth;
    const h = 190;

    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(28, w / h, 0.1, 50);
    this.camera.position.set(3.2, 2.8, 4.2);

    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enablePan = false;
    this.controls.enableZoom = false;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minPolarAngle = Math.PI * 0.15;
    this.controls.maxPolarAngle = Math.PI * 0.65;
    this.controls.target.set(0, 0.2, 0);
    this.controls.update();

    // Lights
    this.scene.add(new THREE.AmbientLight(0x404060, 2.0));
    const dir = new THREE.DirectionalLight(0xccccff, 0.5);
    dir.position.set(3, 5, 4);
    this.scene.add(dir);

    this.buildSubject();
    this.buildHemisphereWire();
    this.buildMarkers();
    this.buildGround();
  }

  private buildSubject() {
    const g = new THREE.Group();

    // Body (capsule)
    const bodyGeo = new THREE.CapsuleGeometry(0.14, 0.3, 4, 8);
    const mat = new THREE.MeshStandardMaterial({
      color: COL.subject, roughness: 0.7, metalness: 0.3,
    });
    const body = new THREE.Mesh(bodyGeo, mat);
    body.position.y = 0.3;
    g.add(body);

    // Head
    const headGeo = new THREE.SphereGeometry(0.11, 12, 12);
    const head = new THREE.Mesh(headGeo, mat);
    head.position.y = 0.62;
    g.add(head);

    // Front indicator (nose cone)
    const noseGeo = new THREE.ConeGeometry(0.035, 0.08, 6);
    const noseMat = new THREE.MeshStandardMaterial({
      color: 0x6666ff, emissive: 0x2222aa,
    });
    const nose = new THREE.Mesh(noseGeo, noseMat);
    nose.position.set(0, 0.62, 0.14);
    nose.rotation.x = Math.PI / 2;
    g.add(nose);

    this.scene.add(g);
  }

  private buildHemisphereWire() {
    const R = HEMISPHERE_RADIUS;

    // Horizontal arcs at each elevation ring
    for (const el of [-25, 0, 30]) {
      const elRad = el * Math.PI / 180;
      const ringR = R * Math.cos(elRad);
      const ringY = R * Math.sin(elRad);
      const pts: THREE.Vector3[] = [];
      for (let a = 0; a <= 180; a += 4) {
        const aRad = a * Math.PI / 180;
        pts.push(new THREE.Vector3(ringR * Math.sin(aRad), ringY, ringR * Math.cos(aRad)));
      }
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.LineBasicMaterial({
        color: COL.wire, transparent: true, opacity: el === 0 ? 0.35 : 0.18,
      });
      this.scene.add(new THREE.Line(geo, mat));
    }

    // Vertical arcs at key azimuths
    for (const az of [0, 45, 90, 135, 180]) {
      const azRad = az * Math.PI / 180;
      const pts: THREE.Vector3[] = [];
      for (let e = -30; e <= 75; e += 4) {
        const eRad = e * Math.PI / 180;
        pts.push(new THREE.Vector3(
          R * Math.cos(eRad) * Math.sin(azRad),
          R * Math.sin(eRad),
          R * Math.cos(eRad) * Math.cos(azRad),
        ));
      }
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.LineBasicMaterial({
        color: COL.wire, transparent: true, opacity: 0.18,
      });
      this.scene.add(new THREE.Line(geo, mat));
    }
  }

  private buildMarkers() {
    for (const preset of CAMERA_PRESETS) {
      const azRad = preset.azimuth * Math.PI / 180;
      const elRad = preset.elevation * Math.PI / 180;
      const R = HEMISPHERE_RADIUS;

      const x = R * Math.cos(elRad) * Math.sin(azRad);
      const y = R * Math.sin(elRad);
      const z = R * Math.cos(elRad) * Math.cos(azRad);

      // Main marker sphere
      const geo = new THREE.SphereGeometry(MARKER_RADIUS, 12, 12);
      const mat = new THREE.MeshStandardMaterial({
        color: COL.default.clone(),
        emissive: COL.defaultEmissive.clone(),
        roughness: 0.3,
        metalness: 0.5,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(x, y, z);
      mesh.userData['preset'] = preset;
      this.scene.add(mesh);
      this.markerMeshes.push(mesh);

      // Glow shell (larger transparent sphere, hidden by default)
      const glowGeo = new THREE.SphereGeometry(MARKER_RADIUS * 2.8, 12, 12);
      const glowMat = new THREE.MeshBasicMaterial({
        color: COL.selected, transparent: true, opacity: 0.12,
      });
      const glow = new THREE.Mesh(glowGeo, glowMat);
      glow.position.copy(mesh.position);
      glow.visible = false;
      this.scene.add(glow);
      this.glowMeshes.set(preset.id, glow);

      // Thin line from subject center to marker
      const lineGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0.4, 0),
        new THREE.Vector3(x, y, z),
      ]);
      const lineMat = new THREE.LineBasicMaterial({
        color: COL.wire, transparent: true, opacity: 0.12,
      });
      this.scene.add(new THREE.Line(lineGeo, lineMat));
    }
  }

  private buildGround() {
    const geo = new THREE.RingGeometry(0.25, HEMISPHERE_RADIUS + 0.1, 48);
    const mat = new THREE.MeshBasicMaterial({
      color: COL.ground, transparent: true, opacity: 0.25, side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(geo, mat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = -0.02;
    this.scene.add(ring);
  }

  // ── Animation Loop ──

  private animate = () => {
    this.animId = requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  // ── Interaction ──

  private bindEvents() {
    const c = this.canvasRef.nativeElement;
    c.addEventListener('pointerdown', this.onPointerDown);
    c.addEventListener('pointermove', this.onPointerMove);
    c.addEventListener('pointerup', this.onPointerUp);
  }

  private onPointerDown = (e: PointerEvent) => {
    this.pointerDownPos = { x: e.clientX, y: e.clientY };
    this.wasDragging = false;
  };

  private onPointerMove = (e: PointerEvent) => {
    // Detect drag (to distinguish from click)
    const dx = e.clientX - this.pointerDownPos.x;
    const dy = e.clientY - this.pointerDownPos.y;
    if (e.buttons > 0 && (Math.abs(dx) > 4 || Math.abs(dy) > 4)) {
      this.wasDragging = true;
    }

    // Raycast for hover
    const rect = this.canvasRef.nativeElement.getBoundingClientRect();
    this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const hits = this.raycaster.intersectObjects(this.markerMeshes);

    // Reset previous hover
    const selectedId = this.presetService.selectedCameraPreset()?.id;
    if (this.hoveredMesh) {
      const pid = (this.hoveredMesh.userData['preset'] as CameraPreset).id;
      if (pid !== selectedId) {
        this.setMarkerVisual(this.hoveredMesh, 'default');
      }
    }

    if (hits.length > 0) {
      const mesh = hits[0].object as THREE.Mesh;
      const pid = (mesh.userData['preset'] as CameraPreset).id;
      if (pid !== selectedId) {
        this.setMarkerVisual(mesh, 'hover');
      }
      this.hoveredMesh = mesh;
      this.canvasRef.nativeElement.style.cursor = 'pointer';
    } else {
      this.hoveredMesh = null;
      this.canvasRef.nativeElement.style.cursor = 'grab';
    }
  };

  private onPointerUp = (_e: PointerEvent) => {
    if (this.wasDragging || !this.hoveredMesh) return;

    const preset = this.hoveredMesh.userData['preset'] as CameraPreset;
    const currentId = this.presetService.selectedCameraPreset()?.id;

    // Toggle: click same again to deselect
    if (currentId === preset.id) {
      this.presetService.selectCameraPosition(null);
      this.setMarkerVisual(this.hoveredMesh, 'hover');
      this.glowMeshes.get(preset.id)!.visible = false;
      this.updateViewRay(null);
      return;
    }

    // Deselect previous
    if (currentId) {
      const prevMesh = this.markerMeshes.find(
        m => (m.userData['preset'] as CameraPreset).id === currentId
      );
      if (prevMesh) this.setMarkerVisual(prevMesh, 'default');
      this.glowMeshes.get(currentId)!.visible = false;
    }

    // Select new
    this.presetService.selectCameraPosition(preset);
    this.setMarkerVisual(this.hoveredMesh, 'selected');
    this.glowMeshes.get(preset.id)!.visible = true;
    this.updateViewRay(this.hoveredMesh);
  };

  // ── Visual Helpers ──

  private setMarkerVisual(mesh: THREE.Mesh, state: 'default' | 'hover' | 'selected') {
    const mat = mesh.material as THREE.MeshStandardMaterial;
    switch (state) {
      case 'default':
        mat.color.copy(COL.default);
        mat.emissive.copy(COL.defaultEmissive);
        mesh.scale.setScalar(1);
        break;
      case 'hover':
        mat.color.copy(COL.hover);
        mat.emissive.copy(COL.hoverEmissive);
        mesh.scale.setScalar(1.3);
        break;
      case 'selected':
        mat.color.copy(COL.selected);
        mat.emissive.copy(COL.selectedEmissive);
        mesh.scale.setScalar(1.5);
        break;
    }
  }

  private updateViewRay(selectedMesh: THREE.Mesh | null) {
    // Remove old
    if (this.viewRay) {
      this.scene.remove(this.viewRay);
      this.viewRay.geometry.dispose();
      (this.viewRay.material as THREE.Material).dispose();
      this.viewRay = null;
    }
    if (!selectedMesh) return;

    // Draw bright line from subject to selected marker
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0.4, 0),
      selectedMesh.position.clone(),
    ]);
    const mat = new THREE.LineBasicMaterial({
      color: COL.primary, transparent: true, opacity: 0.6,
    });
    this.viewRay = new THREE.Line(geo, mat);
    this.scene.add(this.viewRay);
  }
}
