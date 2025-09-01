window.utilCg = (function(){
  function clerpUUID(d){
    // Handle case where d.featureId is undefined
    if (!d.featureId) return '🤖' + d.feature;
    
    // Split featureId by underscore and take first two parts
    const parts = d.featureId.split('_');
    return '🤖' + parts[0] + '_' + parts[1];
  }

  function parseClerpUUID(str){
    var [featureIndex] = str.split('🤖')
    return {featureIndex}
  }

  function loadDatapath(urlStr){
    try {
      var url = new URL(urlStr)
      urlStr = url.searchParams.get('datapath') ?? urlStr
    } catch {}
    urlStr = urlStr?.replace('index.html', 'data.json').split('?')[0] || 'data.json'

    try {
      return util.getFile(urlStr)
    } catch (exc) {
      d3.select('body')
        .html(`Couldn't load data from <code>${urlStr}</code>: ${exc}. Maybe you need to specify a <code>?datapath=</code> argument?`)
        .st({color: '#c00', fontSize: '150%', padding: '1em', whiteSpace: 'pre-wrap'})
      throw exc
    }
  }

  function saveHClerpsToLocalStorage(hClerps, scan) {
    const key = `local-clerp-${scan}`
    const hClerpArray = Array.from(hClerps.entries()).filter(d => d[1])
    localStorage.setItem(key, JSON.stringify(hClerpArray));
  }

  function getHClerpsFromLocalStorage(scan) {
    const key = `local-clerp-${scan}`
    // We want to set on load here so that any page load will fix the key.
    if (localStorage.getItem(key) === null) localStorage.setItem(key, '[]')
    const hClerpArray = JSON.parse(localStorage.getItem(key))
      .filter(d => d[0] != clerpUUID({}))
    return new Map(hClerpArray)
  }

  // if we have multiple transcoder (not just one scan), loads all their clerps into one map
  function getAllHClerpsFromLocalStorage(d) {
    if (d.metadata.scan?.startsWith('custom-')){
      // iterate over all layers
      const transcoder_hclerps = d.metadata.transcoder_list.map(id => getHClerpsFromLocalStorage(id))
      return new Map(function*() { 
        for (const map of transcoder_hclerps) {
          yield* map;
        }
      }());
    } else {
      return getHClerpsFromLocalStorage(d.metadata.scan)
    }
  }

  async function deDupHClerps(scan) {
    const remoteClerps = []
    let remoteMap = new Map(remoteClerps.map(d => {
      let key = clerpUUID(d);
      let clerp = d.clerp;
      return [key, clerp];
    }));

    let localClerps = getHClerpsFromLocalStorage(scan)
    let featureLookup = {}
    data.features.forEach(d => featureLookup[clerpUUID(d)] = d)

    // update feature data with current spreadsheet
    // (why is this behind the "copy" button?)
    Array.from(remoteMap).forEach(([key, value]) => {
      if (featureLookup[key]) featureLookup[key].remoteClerp = value
    })

    const deDupArray = Array.from(localClerps)
      .filter(([key, localClerp]) => {
        let remote = remoteMap.get(key);

        // keep only local clerps that are different from remote
        if (!remote) return true
        // gdoc to local storage mangles quotes, don't force strict equality
        function slugify(d){ return d ? d.replace(/['"]/g, '').trim() : ''}
        if (slugify(remote) != slugify(localClerp)) return true

        // if local changes are on remote, delete localClerp and set remoteClerp
        localClerps.delete(key)
        if (featureLookup[key]) featureLookup[key].localClerp = ''
      })

    // copy feature.remoteClerp to node.remoteClerp
    data.nodes?.forEach(node => {
      var feature = data.features.idToFeature[node.featureId]
      node.remoteClerp = feature.remoteClerp
      node.localClerp = feature.localClerp
    })

    saveHClerpsToLocalStorage(new Map(deDupArray), scan)
    return new Map(deDupArray);
  }

  function tabifyHClerps(hClerps) {
    return []
  }

  function hClerpUpdateFn(params, data){
    // update the relevant clerp
    if (params) {
      const [node, hClerp] = params;
      const scan = data.metadata.scan?.startsWith('custom-') ? data.metadata.transcoder_list[node.layer] : data.metadata.scan;
      console.log(`Updating clerp for node ${node.featureId} in scan ${scan}:`, hClerp)
      const localClerps = getHClerpsFromLocalStorage(scan);
      localClerps.set(clerpUUID(node), hClerp)
      saveHClerpsToLocalStorage(localClerps, scan);
    }

    // load all clerps
    const allLocalClerps = getAllHClerpsFromLocalStorage(data);

    data.features.forEach(node => {
      node.localClerp = allLocalClerps.get(clerpUUID(node))
      node.ppClerp = node.localClerp || node.remoteClerp || node.clerp;
    })

    data.nodes?.forEach(node => {
      var feature = data.features.idToFeature[node.featureId]
      if (!feature) return
      node.localClerp = feature.localClerp
      node.ppClerp = feature.ppClerp
    })

    // Save clerps to url param
    const currentFeatureIds = new Set(data.features.map(d => d.featureIndex.toString()))
    const clerps = Array.from(allLocalClerps.entries())
      .map(([key, value]) => [key.split('🤖')[1].split('_')[1], value])
      .filter(d => currentFeatureIds.has(d[0]))
    util.params.set('clerps', JSON.stringify(clerps))
  }

  // Adds virtual logit node showing A-B logit difference based on url param logitDiff=⍽tokenA⍽__vs__⍽tokenB⍽
  function addVirtualDiff(data){
    // Filter out any previous virtual nodes/links
    var nodes = data.nodes.filter(d => !d.isJsVirtual)
    var links = data.links.filter(d => !d.isJsVirtual)
    nodes.forEach(d => d.logitToken = d.clerp?.split(`"`)[1]?.split(`" k(p=`)[0])

    var [logitAStr, logitBStr] = util.params.get('logitDiff')?.split('__vs__') || []
    if (!logitAStr || !logitBStr) return {nodes, links}
    var logitANode = nodes.find(d => d.logitToken == logitAStr)
    var logitBNode = nodes.find(d => d.logitToken == logitBStr)
    if (!logitANode || !logitBNode) return {nodes, links}

    var virtualId = `virtual-diff-${logitAStr}-vs-${logitBStr}`
    var diffNode = {
      ...logitANode,
      node_id: virtualId,
      jsNodeId: virtualId,
      feature: virtualId,
      isJsVirtual: true,
      logitToken: `${logitAStr} - ${logitBStr}`,
      clerp: `Logit diff: ${logitAStr} - ${logitBStr}`,
    }
    nodes.push(diffNode)

    var targetLinks = links.filter(d => d.target == logitANode.node_id || d.target == logitBNode.node_id)
    d3.nestBy(targetLinks, d => d.source).map(sourceLinks => {
      var linkA = sourceLinks.find(d => d.target == logitANode.node_id)
      var linkB = sourceLinks.find(d => d.target == logitBNode.node_id)

      links.push({
        source: sourceLinks[0].source,
        target: diffNode.node_id,
        weight: (linkA?.weight || 0) - (linkB?.weight || 0),
        isJsVirtual: true
      })
    })

    return {nodes, links}
  }

  function replaceWhitespace(token) {
    return token.replaceAll("\n", "⏎").replaceAll("\t", "→").replaceAll("\r", "↵")
  }

  // Decorates and mutates data.json
  // - Adds pointers between node and links
  // - Deletes very common features
  // - Adds data.features and data.byStream
  async function formatData(data, visState){
    var {metadata} = data
    var {nodes, links} = addVirtualDiff(data)

    var py_node_id_to_node = {}
    var idToNode = {}
    var maxLayer = d3.max(nodes.filter(d => d.feature_type != 'logit'), d => +d.layer)

    if (visState.pruningThreshold) {
      nodes = nodes.filter(d => 
        d.feature_type === 'embedding' || 
        d.feature_type === 'logit' || 
        d.influence <= parseFloat(visState.pruningThreshold) ||
        visState.pinnedIds.includes(d.nodeId)
      )
    }

    // update all prompt tokens to replace \n, \t, \r with ⏎, →, ↵
    data.metadata.prompt_tokens = data.metadata.prompt_tokens.map(replaceWhitespace)

    nodes.forEach((d, i) => {
      // To make hover state work across prompts, drop ctx from node id
      d.featureId = `${d.layer}_${d.feature}_${d.ctx_idx}`
      if (visState.clerps && visState.clerps.has(`${d.feature}`)) {
        d.clerp = visState.clerps.get(`${d.feature}`)
      }

      d.active_feature_idx = d.feature
      d.nodeIndex = i

      if (d.feature_type == 'logit' && !isNaN(maxLayer)) d.layer = maxLayer + 1
      
      // TODO: does this handle error nodes correctly?
      if (d.feature_type == 'unexplored node' && !d.layer != 'E'){
        d.feature_type = 'cross layer transcoder'
      }
      
      // count from end to align last token on diff prompts
      d.ctx_from_end = data.metadata.prompt_tokens.length - d.ctx_idx

      // add clerp to embed and error nodes
      if (d.feature_type.includes('error')){
        d.isError = true

        if (!d.featureId.includes('__err_idx_')) d.featureId = d.featureId + '__err_idx_' + d.ctx_from_end

        if (d.feature_type == 'mlp reconstruction error' && !d.clerp) {
          d.clerp = `Err: mlp “${util.ppToken(data.metadata.prompt_tokens[d.ctx_idx])}"`
        }
      } else if (d.feature_type == 'embedding'){
        d.clerp = `Emb: “${util.ppToken(data.metadata.prompt_tokens[d.ctx_idx])}"`
      }

      d.url = d.vis_link
      d.isFeature = true

      d.targetLinks = []
      d.sourceLinks = []

      // TODO: switch to featureIndex in graphgen
      d.featureIndex = d.feature
      d.nodeId = d.node_id
      if (d.feature_type == 'logit' && d.clerp) d.logitPct= +d.clerp.split('(p=')[1].split(')')[0]
      idToNode[d.nodeId] = d
      py_node_id_to_node[d.node_id] = d
    })

    // delete features that occur in than 2/3 of tokens
    // TODO: more principled way of filtering them out — maybe by feature density?
    var deletedFeatures = []
    var byFeatureId = d3.nestBy(nodes, d => d.featureId)
    byFeatureId.forEach(feature => {
      if (feature.length > metadata.prompt_tokens.length*2/3){
        deletedFeatures.push(feature)
        feature.forEach(d => {
          delete idToNode[d.nodeId]
          delete py_node_id_to_node[d.node_id]
        })
      }
    })
    if (deletedFeatures.length) console.log({deletedFeatures})
    nodes = nodes.filter(d => idToNode[d.nodeId])
    nodes = d3.sort(nodes, d => +d.layer)

    links = links.filter(d => py_node_id_to_node[d.source] && py_node_id_to_node[d.target])

    // connect links to nodes
    links.forEach(link => {
      link.sourceNode = py_node_id_to_node[link.source]
      link.targetNode = py_node_id_to_node[link.target]

      link.linkId = link.sourceNode.nodeId + '__' + link.targetNode.nodeId

      link.sourceNode.targetLinks.push(link)
      link.targetNode.sourceLinks.push(link)
      link.absWeight = Math.abs(link.weight)
    })
    links = d3.sort(links, d => d.absWeight) 
    

    nodes.forEach(d => {
      d.inputAbsSum = d3.sum(d.sourceLinks, e => Math.abs(e.weight))
      d.sourceLinks.forEach(e => e.pctInput = e.weight/d.inputAbsSum)
      d.inputError = d3.sum(d.sourceLinks.filter(e => e.sourceNode.isError), e => Math.abs(e.weight))
      d.pctInputError = d.inputError/d.inputAbsSum
    })

    // convert layer/probe_location_idx to a streamIdx used to position nodes
    var byStream = d3.nestBy(nodes, d => [d.layer, d.probe_location_idx] + '')
    byStream = d3.sort(byStream, d => d[0].probe_location_idx)
    byStream = d3.sort(byStream, d => d[0].layer == 'E' ? -1 : +d[0].layer)
    byStream.forEach((stream, streamIdx) => {
      stream.forEach(d => {
        d.streamIdx = streamIdx
        d.layerLocationLabel = layerLocationLabel(d.layer, d.probe_location_idx)

        if (!visState.isHideLayer) d.streamIdx = isFinite(d.layer) ? +d.layer + 1 : 0
      })
    })

    // add target_logit_effect__ columns for each logit
    var logitNodeMap = new Map(nodes.filter(d => d.isLogit).map(d => [d.node_id, d.logitToken]))
    nodes.forEach(node => {
      node.targetLinks.forEach(link => {
        if (!logitNodeMap.has(link.target)) return
        node[`target_logit_effect__${logitNodeMap.get(link.target)}`] = link.weight
      })
    })

    // add ppClerp
    await Promise.all(nodes.map(async d => {
      if (!d.clerp) d.clerp = ''
      d.remoteClerp = ''
    }))

    // condense nodes into features, using last occurence of feature if necessary to point to a node
    var features = d3.nestBy(nodes.filter(d => d.isFeature), d => d.featureId)
      .map(d => ({
        featureId: d[0].featureId,
        feature_type: d[0].feature_type,
        clerp: d[0].clerp,
        remoteClerp: d[0].remoteClerp,
        layer: d[0].layer,
        streamIdx: d[0].streamIdx,
        probe_location_idx: d[0].probe_location_idx,
        featureIndex: d[0].featureIndex,
        top_logit_effects: d[0].top_logit_effects,
        bottom_logit_effects: d[0].bottom_logit_effects,
        top_embedding_effects: d[0].top_embedding_effects,
        bottom_embedding_effects: d[0].bottom_embedding_effects,
        url: d[0].url,
        lastNodeId: d.at(-1).nodeId,
        isLogit: d[0].isLogit,
        isError: d[0].isError,
        feature_type: d[0].feature_type,
      }))

    nodes.idToNode = idToNode
    features.idToFeature = Object.fromEntries(features.map(d => [d.featureId, d]))
    links.idToLink = Object.fromEntries(links.map(d => [d.linkId, d]))

    // Create a copy of the data object with the new properties
    return { ...data, nodes, features, links, byStream };
  }

  function initBcSync({visState, renderAll}){
    var bcStateSync = window.bcSync = window.bcSync ||  new BroadcastChannel('state-sync')

    function broadcastState(){
      if (!visState.isSyncEnabled) return
      bcStateSync.postMessage({
        pinnedIds: visState.pinnedIds,
        hiddenIds: visState.hiddenIds,
        clickedId: visState.clickedId,
        hoveredId: visState.hoveredId,
        pageUUID: visState.pageUUID,
        isSyncEnabled: visState.isSyncEnabled,
      })
    }

    renderAll.pinnedIds.fns.push(ev => { if (!ev?.skipBroadcast) broadcastState() })
    renderAll.hiddenIds.fns.push(ev => { if (!ev?.skipBroadcast) broadcastState() })
    renderAll.clickedId.fns.push(ev => { if (!ev?.skipBroadcast) broadcastState() })
    renderAll.hoveredId.fns.push(ev => { if (!ev?.skipBroadcast) broadcastState() })

    bcStateSync.onmessage = ev => {
      if (!visState.isSyncEnabled) return
      if (visState.isSyncEnabled != ev.data.isSyncEnabled) return
      if (ev.data.pageUUID == visState.pageUUID) return

      if (JSON.stringify(visState.pinnedIds) != JSON.stringify(ev.data.pinnedIds)){
        visState.pinnedIds = ev.data.pinnedIds
        renderAll.pinnedIds({skipBroadcast: true})
      }

      if (JSON.stringify(visState.hiddenIds) != JSON.stringify(ev.data.hiddenIds)){
        visState.hiddenIds = ev.data.hiddenIds
        renderAll.hiddenIds({skipBroadcast: true})
      }

      if (visState.clickedId != ev.data.clickedId){
        visState.clickedId = ev.data.clickedId
        renderAll.clickedId({skipBroadcast: true})
      }

      if (visState.hoveredId != ev.data.hoveredId){
        visState.hoveredId = ev.data.hoveredId
        renderAll.hoveredId({skipBroadcast: true})
      }
    }
  }

  function addFeatureEvents(visState, renderAll) {
    return function(selection) {
      selection
        .on('mouseover', (ev, d) => {
          if (ev.shiftKey) return
          if (visState.subgraph?.activeGrouping.isActive) return
          if (visState.subgraph?.activeGrouping.isDragging) return
          ev.preventDefault()
          hoverFeature(visState, renderAll, d)
        })
        .on('mouseout', (ev, d) => {
          if (ev.shiftKey) return
          if (visState.subgraph?.activeGrouping.isActive) return
          if (visState.subgraph?.activeGrouping.isDragging) return
          ev.preventDefault()
          unHoverFeature(visState, renderAll)
        })
        .on('click', (ev, d) => {
          if (visState.subgraph?.activeGrouping.isActive) return
          clickFeature(visState, renderAll, d, ev.metaKey || ev.ctrlKey)
        })
    }
  }

  function hoverFeature(visState, renderAll, d) {
    if (d.nodeId.includes('supernode-')) return

    if (visState.hoveredId != d.featureId) {
      visState.hoveredId = d.featureId
      visState.hoveredCtxIdx = d.ctx_idx
      renderAll.hoveredId()
    }
  }

  function unHoverFeature(visState, renderAll) {
    if (visState.hoveredId) {
      visState.hoveredId = null
      visState.hoveredCtxIdx = null
      setTimeout(() => {
        if (!visState.hoveredId) renderAll.hoveredId()
      })
    }
  }
  function togglePinned(visState, renderAll, d) {
    var index = visState.pinnedIds.indexOf(d.nodeId)
    if (index == -1) {
      visState.pinnedIds.push(d.nodeId)
    } else {
      visState.pinnedIds.splice(index, 1)
    }
    util.params.set('pinnedIds', visState.pinnedIds.join(','))
    renderAll.pinnedIds()
  }

  function clickFeature(visState, renderAll, d, metaKey){
    console.log(d)
    if (d.nodeId.includes('supernode-')) return
    
    if (metaKey && visState.isEditMode) {
      togglePinned(visState, renderAll, d) 
    } else {
      if (visState.clickedId == d.nodeId) {
        visState.clickedId = null
        visState.clickedCtxIdx = null
      } else {
        visState.clickedId = d.nodeId
        visState.clickedCtxIdx = d.ctx_idx
      }
      visState.hoveredId = null
      visState.hoveredCtxIdx = null
      renderAll.clickedId()
    }
  }

  function showTooltip(ev, d) {
    let tooltipSel = d3.select('.tooltip'),
        x = ev.clientX,
        y = ev.clientY,
        bb = tooltipSel.node().getBoundingClientRect(),
        left = d3.clamp(20, (x-bb.width/2), window.innerWidth - bb.width - 20),
        top = innerHeight > y + 20 + bb.height ? y + 20 : y - bb.height - 20;

    let tooltipHtml = !ev.metaKey ? (d.ppClerp || `F#${d.feature}`) : Object.keys(d)
      .filter(str => typeof d[str] != 'object' && typeof d[str] != 'function' && !keysToSkip.has(str))
      .map(str => {
        var val = d[str]
        if (typeof val == 'number' && !Number.isInteger(val)) val = val.toFixed(6)
        return `<div>${str}: <b>${val}</b></div>`
      })
      .join('')

    tooltipSel
      .style('left', left +'px')
      .style('top', top + 'px')
      .html(tooltipHtml)
      .classed('tooltip-hidden', false)
  }

  function addFeatureTooltip(selection){
    selection
      .call(d3.attachTooltip, d3.select('.tooltip'), [])
      .on('mouseover.tt', (ev, d) => {
        var tooltipHtml = !ev.metaKey ? d.ppClerp : Object.keys(d)
          .filter(str => typeof d[str] != 'object' && typeof d[str] != 'function' && !keysToSkip.has(str))
          .map(str => {
            var val = d[str]
            if (typeof val == 'number' && !Number.isInteger(val)) val = val.toFixed(6)
            return `<div>${str}: <b>${val}</b></div>`
          })
          .join('')

        d3.select('.tooltip').html(tooltipHtml)
      })
  }

  function hideTooltip() {
    d3.select('.tooltip').classed('tooltip-hidden', true);
  }

  function updateFeatureStyles(nodeSel){
    nodeSel.call(classAndRaise('hovered', e => e.featureId == visState.hoveredId))

    var pinnedIdSet = new Set(visState.pinnedIds)
    nodeSel.call(classAndRaise('pinned', d => pinnedIdSet.has(d.nodeId)))

    var hiddenIdSet = new Set(visState.hiddenIds)
    nodeSel.call(classAndRaise('hidden', d => hiddenIdSet.has(d.featureId)))

    if (nodeSel.datum().nodeId){
      nodeSel.call(classAndRaise('clicked', e => e.nodeId === visState.clickedId))
    } else {
      nodeSel.call(classAndRaise('clicked', d => d.featureId == visState.clickedId))
    }
  }

  function classAndRaise(className, filterFn) {
    return sel => {
      sel
        .classed(className, 0)
        .filter(filterFn)
        .classed(className, 1)
        .raise()
    }
  }

  var keysToSkip = new Set([
    'node_id', 'jsNodeId', 'nodeId', 'layerLocationLabel', 'remoteClerp', 'localClerp', 
    'tmpClickedTargetLink', 'tmpClickedLink', 'tmpClickedSourceLink',
    'pos', 'xOffset', 'yOffset', 'sourceLinks', 'targetLinks', 'url', 'vis_link', 'run_idx',
    'featureId', 'active_feature_idx', 'nodeIndex', 'isFeature', 'Distribution',
    'clerp', 'ppClerp', 'is_target_logit', 'token_prob', 'reverse_ctx_idx', 'ctx_from_end', 'feature', 'logitToken',
    'featureIndex', 'streamIdx', 'nodeColor', 'umap_enc_x', 'umap_enc_y', 'umap_dec_x', 'umap_dec_y', 'umap_concat_x', 'umap_concat_y',
  ])
  

  function layerLocationLabel(layer, location) {
    if (layer == 'E') return 'Emb'
    if (layer == 'E1') return 'Lgt'
    if (location === -1) return 'logit'

    // TODO: is stream probe_location_idx no longer be saved out?
    // NOTE: For now, location is literally ProbePointLocation
    return `L${layer}`
  }

  var memoize = fn => {
    var cache = new Map()
    return (...args) => {
      var key = JSON.stringify(args)
      if (cache.has(key)) return cache.get(key)
      var result = fn(...args)
      cache.set(key, result)
      return result
    }
  }

  var bgColorToTextColor = memoize((backgroundColor, light='#fff', dark='#000') => {
    if (!backgroundColor) return ''
    var hsl = d3.hsl(backgroundColor)
    return hsl.l > 0.55 ? dark : light
  })

  // gradient for hover && pinned state
  function addPinnedClickedGradient(svg){
    svg.append('defs').html(`
      <linearGradient id='pinned-clicked-gradient' x1='0' x2='2' gradientUnits='userSpaceOnUse' spreadMethod='repeat'>
        <stop offset='0'    stop-color='#f0f' />
        <stop offset='70%'  stop-color='#f0f' />
        <stop offset='71%'  stop-color='#000' />
        <stop offset='100%' stop-color='#000' />
      </linearGradient>
    `)
  }

  function renderFeatureRow(sel, visState, renderAll, linkKey='tmpClickedLink'){
    sel.st({
      background: d => d[linkKey]?.tmpColor,
      color: d => bgColorToTextColor(d[linkKey]?.tmpColor, '#eee', '#555'),
    })

    // add events in a timeout to avoid connection clicks leading to an instant hover
    setTimeout(() => sel.call(addFeatureEvents(visState, renderAll)), 16)

    let featureIconSel = sel.append('svg')
      .at({width: 10, height: 10})

    let featureIcon = featureIconSel.append('g')

    featureIcon.append('g.default-icon').append('text')
      .text(d => featureTypeToText(d.feature_type))
      .at({
        fontSize: 9,
        textAnchor: 'middle',
        dominantBaseline: 'central',
        dx: 5,
        dy: 4,
      })
      .at({fill: d => d[linkKey]?.tmpColor})


    sel.append('div.label')
      .text(d => d.ppClerp)
      .at({ title: d => d.ppClerp })

    sel
      .filter(d => d[linkKey] && d[linkKey].tmpClickedCtxOffset != 0)
      .append('div.ctx-offset')
      .text(d => d[linkKey].tmpClickedCtxOffset < 0 ? '←' : '→')

    if (!visState.isHideLayer){
      sel.append('div.layer')
        .text(d => d.layerLocationLabel ?? layerLocationLabel(d.layer, d.probe_location_idx));
    }

    sel.append('div.weight')
      .text(d => d[linkKey] ? d3.format('+.3f')(d[linkKey].weight) : '')
  }

  function featureTypeToText(type){
    if (type == 'logit') return '■'
    if (type == 'embedding') return '■'
    if (type === 'mlp reconstruction error') return '◆'
    return '●'
    
  }


  return {
    loadDatapath,
    formatData,
    initBcSync,
    addFeatureEvents,
    hoverFeature,
    unHoverFeature,
    clickFeature,
    togglePinned,
    layerLocationLabel,
    keysToSkip,
    addFeatureTooltip,
    showTooltip,
    hideTooltip,
    updateFeatureStyles,
    memoize,
    bgColorToTextColor,
    addPinnedClickedGradient,
    renderFeatureRow,
    saveHClerpsToLocalStorage,
    getHClerpsFromLocalStorage,
    hClerpUpdateFn,
    deDupHClerps,
    tabifyHClerps,
    featureTypeToText,
  }
})()

window.init?.()
