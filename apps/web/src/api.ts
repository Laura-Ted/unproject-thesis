/**
 * Everything the viewer fetches.
 *
 * The API is the only thing that knows what a run is; this file just parses
 * what it says. Responses are validated rather than trusted, so a schema drift
 * fails here instead of drawing something plausible and wrong.
 */

import { z } from 'zod';

import type { BodiesArtifact, BodyTopology } from './bodies';
import type { BodySource, RestBody } from './bodies';
import { bodiesArtifactSchema, transposeMatrices } from './bodies';

const runSummarySchema = z.object({
  path: z.string(),
  experiment: z.string(),
  hash: z.string(),
  frames: z.number().int().nonnegative(),
  people: z.number().int().nonnegative(),
  max_people: z.number().int().nonnegative(),
  tracks: z.number().int().nonnegative(),
  duration_seconds: z.number().nonnegative(),
  // The footage this run was fitted to, when it is still on disk.
  video: z.string().nullable(),
});

export type RunSummary = z.infer<typeof runSummarySchema>;

const uploadSchema = z.object({ path: z.string(), name: z.string() });
export type Upload = z.infer<typeof uploadSchema>;

async function readJson(url: string, init?: RequestInit): Promise<unknown> {
  const response = await fetch(url, init);
  if (!response.ok) {
    // The API puts its reason in `detail`; showing it beats "500".
    const detail = await response.text().catch(() => '');
    throw new Error(detail.slice(0, 400) || `${url} returned ${String(response.status)}`);
  }
  return response.json();
}

export async function listRuns(): Promise<RunSummary[]> {
  return z.array(runSummarySchema).parse(await readJson('/api/runs'));
}

export async function uploadVideo(file: File): Promise<Upload> {
  const body = new FormData();
  body.append('file', file);
  return uploadSchema.parse(await readJson('/api/uploads', { method: 'POST', body }));
}

/**
 * Remove a finished run from disk. Irreversible.
 *
 * The server checks that the path is a bodies run inside the artifact tree, so
 * this cannot reach anything else; the caller is responsible for asking the
 * operator first, because the artifact is the evidence behind any number drawn
 * from it. ADR-0002 §10.5.
 */
export async function deleteRun(path: string): Promise<void> {
  const response = await fetch(`/api/runs/${path}`, { method: 'DELETE' });
  if (!response.ok) {
    throw new Error(`could not remove the run (${String(response.status)})`);
  }
}

export async function loadArtifact(path: string): Promise<BodiesArtifact> {
  return bodiesArtifactSchema.parse(await readJson(`/artifacts/${path}/bodies.json`));
}

/**
 * Fetch one array out of the parameter sidecar.
 *
 * A range request: drawing skeletons needs the joints, which are a twelfth of
 * what the file weighs.
 */
export async function loadArray(
  path: string,
  artifact: BodiesArtifact,
  name: string,
): Promise<Float32Array> {
  const array = artifact.parameters.arrays.find((entry) => entry.name === name);
  if (array === undefined) throw new Error(`the sidecar holds no array named ${name}`);

  const start = array.byte_offset;
  const end = start + array.byte_length - 1;
  const response = await fetch(`/artifacts/${path}/${artifact.parameters.path}`, {
    headers: { Range: `bytes=${String(start)}-${String(end)}` },
  });
  if (!response.ok) {
    throw new Error(`${artifact.parameters.path} returned ${String(response.status)}`);
  }

  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== array.byte_length) {
    // A server that ignored the range hands back the whole file, and reading
    // that as though it were the slice would be silently wrong.
    throw new Error(
      `expected ${String(array.byte_length)} bytes of ${name}, got ${String(buffer.byteLength)}`,
    );
  }
  return new Float32Array(buffer);
}

export interface MeshData {
  readonly faces: Uint16Array;
  readonly skinIndices: Uint16Array;
  readonly skinWeights: Float32Array;
  /** (tracks, vertices, 3) laid out flat. */
  readonly restVertices: Float32Array;
  /** (tracks, bones, 16) laid out flat. */
  readonly restBones: Float32Array;
  readonly vertexCount: number;
  readonly boneCount: number;
  /** Which track each rest row belongs to. */
  readonly tracks: readonly number[];
}

/**
 * Fetch the body: topology, skinning, and a rest pose per track.
 *
 * One request for the whole file. It is a few megabytes and every part of it is
 * needed, so ranging it into five would only add round trips.
 */
export async function loadMesh(
  path: string,
  artifact: BodiesArtifact,
  boneCount: number,
): Promise<MeshData> {
  const mesh = artifact.mesh;
  if (mesh === null) throw new Error('this run has no mesh; it can only be drawn as a skeleton');

  const response = await fetch(`/artifacts/${path}/${mesh.path}`);
  if (!response.ok) throw new Error(`${mesh.path} returned ${String(response.status)}`);
  const buffer = await response.arrayBuffer();

  const slice = (name: string): { buffer: ArrayBuffer; count: number; dtype: string } => {
    const entry = mesh.arrays.find((array) => array.name === name);
    if (entry === undefined) throw new Error(`the mesh holds no array named ${name}`);
    const size = entry.dtype === '<f4' ? 4 : 2;
    return {
      buffer: buffer.slice(entry.byte_offset, entry.byte_offset + entry.byte_length),
      count: entry.byte_length / size,
      dtype: entry.dtype,
    };
  };
  const shorts = (name: string): Uint16Array => new Uint16Array(slice(name).buffer);
  const floats = (name: string): Float32Array => new Float32Array(slice(name).buffer);

  return {
    faces: shorts('faces'),
    skinIndices: shorts('skin_indices'),
    skinWeights: floats('skin_weights'),
    restVertices: floats('rest_vertices'),
    // Column-major from here on, which is what three.js reads.
    restBones: transposeMatrices(floats('rest_bone_poses')),
    vertexCount: mesh.vertex_count,
    boneCount,
    tracks: mesh.tracks,
  };
}

/** A finished run's mesh, as the renderer wants it. */
export function bodySourceFromMesh(mesh: MeshData): BodySource {
  const rows = new Map(mesh.tracks.map((track, row) => [track, row]));
  const vertexStride = mesh.vertexCount * 3;
  const boneStride = mesh.boneCount * 16;
  return {
    topology: {
      faces: mesh.faces,
      skinIndices: mesh.skinIndices,
      skinWeights: mesh.skinWeights,
      vertexCount: mesh.vertexCount,
      boneCount: mesh.boneCount,
    },
    rest(track: number): RestBody | null {
      const row = rows.get(track);
      if (row === undefined) return null;
      return {
        vertices: mesh.restVertices.subarray(row * vertexStride, (row + 1) * vertexStride),
        bones: mesh.restBones.subarray(row * boneStride, (row + 1) * boneStride),
      };
    },
  };
}

const topologySchema = z.object({
  vertex_count: z.number().int().positive(),
  face_count: z.number().int().positive(),
  influences: z.number().int().positive(),
  bone_count: z.number().int().positive(),
  arrays: z.array(
    z.object({
      name: z.string(),
      dtype: z.enum(['<f4', '<u2']),
      shape: z.array(z.number().int().nonnegative()),
      byte_offset: z.number().int().nonnegative(),
      byte_length: z.number().int().nonnegative(),
    }),
  ),
});

/**
 * The body's topology, for a stream.
 *
 * Constant for the model, so it is one request rather than something the socket
 * repeats. A finished run carries its own copy in the artifact.
 */
export async function loadTopology(): Promise<BodyTopology> {
  const described = topologySchema.parse(await readJson('/api/body/topology'));
  const response = await fetch('/api/body/topology.bin');
  if (!response.ok) throw new Error(`the body topology returned ${String(response.status)}`);
  const buffer = await response.arrayBuffer();

  const cut = (name: string): ArrayBuffer => {
    const entry = described.arrays.find((array) => array.name === name);
    if (entry === undefined) throw new Error(`the topology holds no ${name}`);
    return buffer.slice(entry.byte_offset, entry.byte_offset + entry.byte_length);
  };
  return {
    faces: new Uint16Array(cut('faces')),
    skinIndices: new Uint16Array(cut('skin_indices')),
    skinWeights: new Float32Array(cut('skin_weights')),
    vertexCount: described.vertex_count,
    boneCount: described.bone_count,
  };
}
