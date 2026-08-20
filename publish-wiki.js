#!/usr/bin/env node
/**
 * publish-wiki.js — 自动生成 index.md（全站内容目录）
 *
 * 用法：node publish-wiki.js
 *
 * 逻辑：递归扫描 wiki/ 下所有 .md（跳过 _ 开头目录），按一级主题目录分组，
 * 输出与仓库现有格式一致的 index.md。显示名取文件名（去 .md 扩展名）。
 * 零依赖，仅用 Node 内置模块。
 */
'use strict';

const fs = require('fs');
const path = require('path');

const WIKI_ROOT = path.join(__dirname, 'wiki');
const INDEX_FILE = path.join(__dirname, 'index.md');

/** 一级主题目录 → index.md 区段名 */
const SECTION_NAMES = {
  'agent-engineering': 'Agent 工程',
  'ai-infra': 'AI 基础设施',
  'IC': 'IC 集成电路',
  'llm': 'LLM 大语言模型',
};

/** 递归收集目录下所有 .md 文件（跳过 _ 开头目录与隐藏文件） */
function collectMarkdown(dir) {
  const files = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true })
    .sort((a, b) => a.name.localeCompare(b.name, 'zh-Hans-CN'));
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name.startsWith('_')) continue; // _assets 等附件目录
      files.push(...collectMarkdown(full));
    } else if (entry.name.endsWith('.md')) {
      files.push(full);
    }
  }
  return files;
}

/** 把绝对路径转成相对仓库根的 POSIX 路径，并对每一段做 URL 编码（空格→%20、中文/全角括号→百分号编码、半角括号→%28/%29） */
function toRepoPath(abs) {
  const rel = path.relative(__dirname, abs).split(path.sep);
  return rel.map((s) => encodeURIComponent(s).replace(/\(/g, '%28').replace(/\)/g, '%29')).join('/');
}

function buildIndex() {
  const tops = fs.readdirSync(WIKI_ROOT, { withFileTypes: true })
    .filter((e) => e.isDirectory() && !e.name.startsWith('_') && !e.name.startsWith('.'))
    .map((e) => e.name)
    .sort((a, b) => a.localeCompare(b, 'zh-Hans-CN'));

  const lines = ['# 内容目录', '', '> 本文件由 `publish-wiki.js` 自动生成，请勿手改。'];

  for (const top of tops) {
    const title = SECTION_NAMES[top] || top;
    lines.push('', `## ${title}`);
    const topDir = path.join(WIKI_ROOT, top);
    const files = collectMarkdown(topDir);
    // 按相对子目录分组（保持目录编号顺序），顶层散文件排在最前
    const groups = new Map();
    for (const f of files) {
      const relDir = path.relative(topDir, path.dirname(f)).split(path.sep).join(' / ');
      const key = relDir || '';
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(f);
    }
    for (const [dir, groupFiles] of groups) {
      if (dir) lines.push('', `### ${dir}`, '');
      else lines.push('');
      for (const f of groupFiles) {
        const name = path.basename(f, '.md');
        lines.push(`- [${name}](${toRepoPath(f)})`);
      }
    }
  }

  return lines.join('\n') + '\n';
}

fs.writeFileSync(INDEX_FILE, buildIndex(), 'utf-8');
console.log(`✅ index.md generated (${WIKI_ROOT})`);
