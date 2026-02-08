---
title: categories
layout: categories
date: 2026-02-06 13:06:41
subtitle: "分门别类，井然有序"  
---

<%
const descMap = (site.data && site.data.categories) ? site.data.categories : {};
const cats = site.categories ? site.categories.toArray() : [];
cats.sort((a,b)=> (b.length||0)-(a.length||0));
%>

<div class="cat-desc-list">
  <% cats.forEach(cat => { %>
    <div class="cat-desc-item">
      <div class="cat-desc-head">
        <a href="<%= url_for(cat.path) %>"><%= cat.name %></a>
        <span class="cat-desc-count"><%= cat.length %></span>
      </div>
      <% if (descMap[cat.name]) { %>
        <div class="cat-desc-quote"><%= descMap[cat.name] %></div>
      <% } %>
    </div>
  <% }) %>
</div>