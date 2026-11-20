/**
 * @file    icaruscode/TPC/NuGraph/ICARUSNuGraphAnalyzerRecoFix_module.cc
 * @brief   Analyzer to check on TPC hits and their relationship with NuGraph2 predictions.
 * @author  Riccardo Triozzi ( triozzi@pd.infn.it ), Giuseppe Cerati ( cerati@fnal.gov )
 * @date    September 9, 2025
 *
 * ============================================================================
 * RECO-ONLY BUGFIX PASS -- see inline "FIX N" comments for each change.
 * No truth-matching, no BackTrackerService/ParticleInventoryService, nothing
 * MC-only. That's a separate, larger pass, kept deliberately out of this file
 * so this diff stays reviewable as "just the pre-existing bugs, fixed."
 *
 * Three issues found in the original, all confirmed by direct inspection of
 * the code that was in production, not speculation:
 *
 * FIX 1 -- _icluster and _ipfp were declared as plain `int` class members
 *   with no in-class initializer, branched onto NuGraphHitOutput, but NEVER
 *   assigned anywhere in analyze(). Every Fill() wrote whatever bit pattern
 *   happened to be sitting in that memory since object construction -- this
 *   is undefined behavior, not just "defaults to 0." Fixed by actually
 *   building the PFParticle->Cluster->Hit association (the standard Pandora
 *   chain -- there's no direct PFParticle->Hit association in the data
 *   model) and assigning both every iteration, with -1 as an explicit,
 *   intentional sentinel for "no cluster/PFP found for this hit."
 *
 * FIX 2 -- _treeEvt was `tfs->make<TTree>(...)` with its branches booked in
 *   the constructor, but analyze() never called _treeEvt->Fill() and never
 *   touched vtx_x/y/z. Run the original as-is and NuGraphEventOutput has
 *   zero entries in the output file, permanently -- a silently dead tree.
 *   Fixed by actually calling Fill() once per event. For vtx_x/y/z itself:
 *   the original already #includes lardataobj/RecoBase/Vertex.h but never
 *   uses it anywhere -- a strong signal the intent was to report Pandora's
 *   *reconstructed* vertex (via a PFParticle->Vertex association), not
 *   truth. So that's what this fixes it to do: the vertex associated with
 *   the first primary PFParticle found that has one. NOTE if an event has
 *   more than one slice with a primary PFP+vertex (e.g. multiple candidate
 *   neutrino slices), only the first one encountered is reported here --
 *   this module has no per-slice event tree, so a single "the" vertex per
 *   event is the best fit for the existing schema. Flag if that assumption
 *   doesn't hold for your samples.
 *
 * FIX 3 -- On the "no valid NuGraph output" branch, _x_filter/_MIP/_HIP/
 *   _shower/_michel/_diffuse were never reset. Since these are class
 *   members (not locals), a row for a hit with no NG output silently
 *   carried the PREVIOUS hit's real scores forward. Fixed by adding a
 *   has_ng_output flag and explicitly resetting all five to -1 on that
 *   branch, so a stale carryover can never be mistaken for a real
 *   prediction downstream.
 * ============================================================================
 */

#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"
#include "canvas/Utilities/InputTag.h"
#include "canvas/Persistency/Common/Assns.h"
#include "canvas/Persistency/Common/FindOneP.h"
#include "canvas/Persistency/Common/FindManyP.h"
#include "fhiclcpp/ParameterSet.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include "TTree.h"
#include "art_root_io/TFileService.h"

#include <limits>
#include <unordered_map>

#include "lardata/RecoBaseProxy/ProxyBase.h"
#include "lardataobj/AnalysisBase/MVAOutput.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/Vertex.h"
#include "lardataobj/RecoBase/Slice.h"
#include "lardataobj/RecoBase/Cluster.h"
#include "lardataobj/RecoBase/PFParticle.h"

class ICARUSNuGraphAnalyzerRecoFix;

using std::vector;

class ICARUSNuGraphAnalyzerRecoFix : public art::EDAnalyzer {
public:
  explicit ICARUSNuGraphAnalyzerRecoFix(fhicl::ParameterSet const& p);

  ICARUSNuGraphAnalyzerRecoFix(ICARUSNuGraphAnalyzerRecoFix const&) = delete;
  ICARUSNuGraphAnalyzerRecoFix(ICARUSNuGraphAnalyzerRecoFix&&) = delete;
  ICARUSNuGraphAnalyzerRecoFix& operator=(ICARUSNuGraphAnalyzerRecoFix const&) = delete;
  ICARUSNuGraphAnalyzerRecoFix& operator=(ICARUSNuGraphAnalyzerRecoFix&&) = delete;

  void analyze(art::Event const& e) override;

private:
  TTree *_treeHit, *_treeEvt;
  int _run, _subrun, _event, _id, _wire, _plane, _tpc, _cryo;
  float _x_filter, _MIP, _HIP, _shower, _michel, _diffuse, _time;
  int _islc, _icluster, _ipfp;
  float _vtx_x, _vtx_y, _vtx_z;
  std::string fNGLabel, fSliceLabel, fPandoraLabel;

  // FIX 3: flag so downstream analysis can tell a real NG prediction from a
  // hit that had none, instead of relying on the five score columns being
  // "nonzero" (which is exactly what let the stale-carryover bug hide).
  int _has_ng_output;
};

ICARUSNuGraphAnalyzerRecoFix::ICARUSNuGraphAnalyzerRecoFix(fhicl::ParameterSet const& p)
  : EDAnalyzer{p},
  fNGLabel{p.get<std::string>("NuGraphLabel", "NGMultiSliceCryoE")},
  fSliceLabel{p.get<std::string>("SliceLabel", "NCCSlicesCryoE")},
  fPandoraLabel{p.get<std::string>("PandoraLabel", "pandoraGausCryoE")}
{
  art::ServiceHandle<art::TFileService> tfs;
  _treeHit = tfs->make<TTree>("NuGraphHitOutput", "NuGraphHitOutput");
  _treeHit->Branch("run", &_run, "run/I");
  _treeHit->Branch("subrun", &_subrun, "subrun/I");
  _treeHit->Branch("event", &_event, "event/I");
  _treeHit->Branch("id", &_id, "id/I");
  _treeHit->Branch("wire", &_wire, "wire/I");
  _treeHit->Branch("plane", &_plane, "plane/I");
  _treeHit->Branch("tpc", &_tpc, "tpc/I");
  _treeHit->Branch("cryo", &_cryo, "cryo/I");
  _treeHit->Branch("x_filter", &_x_filter, "x_filter/F");
  _treeHit->Branch("MIP", &_MIP, "MIP/F");
  _treeHit->Branch("HIP", &_HIP, "HIP/F");
  _treeHit->Branch("shower", &_shower, "shower/F");
  _treeHit->Branch("michel", &_michel, "michel/F");
  _treeHit->Branch("diffuse", &_diffuse, "diffuse/F");
  _treeHit->Branch("time", &_time, "time/F");
  _treeHit->Branch("islc", &_islc, "islc/I");
  // FIX 1: these two now correspond to a real association walk below, and
  // are explicitly -1 (not garbage) when a hit has no cluster/PFP.
  _treeHit->Branch("icluster", &_icluster, "icluster/I");
  _treeHit->Branch("ipfp", &_ipfp, "ipfp/I");
  // FIX 3
  _treeHit->Branch("has_ng_output", &_has_ng_output, "has_ng_output/I");

  _treeEvt = tfs->make<TTree>("NuGraphEventOutput", "NuGraphEventOutput");
  _treeEvt->Branch("run", &_run, "run/I");
  _treeEvt->Branch("subrun", &_subrun, "subrun/I");
  _treeEvt->Branch("event", &_event, "event/I");
  _treeEvt->Branch("vtx_x", &_vtx_x, "vtx_x/F");
  _treeEvt->Branch("vtx_y", &_vtx_y, "vtx_y/F");
  _treeEvt->Branch("vtx_z", &_vtx_z, "vtx_z/F");
}

void ICARUSNuGraphAnalyzerRecoFix::analyze(art::Event const& e)
{
  // Get slices. (unchanged from original)
  const std::vector<art::Ptr<recob::Slice>> slices = e.getProduct<vector<art::Ptr<recob::Slice>>>(fSliceLabel);
  art::FindManyP<recob::Hit> findMHitsFromSlice(slices, e, fPandoraLabel);

  // Map hits to slices. (unchanged from original)
  std::vector<art::Ptr<recob::Hit>> allHits;
  std::map<unsigned int, int> hitToSliceID;
  for (size_t islc = 0; islc < slices.size(); ++islc) {
    art::Ptr<recob::Slice> slice = slices[islc];
    const std::vector<art::Ptr<recob::Hit>> hitsInSlice = findMHitsFromSlice.at(islc);
    for (auto const& h : hitsInSlice) {
      hitToSliceID[h.key()] = slice.key();
      allHits.emplace_back(h);
    }
  }

  // ---- FIX 1: PFParticle->Cluster->Hit association (reco-only) ----
  // No direct PFParticle->Hit association exists in the data model; Cluster
  // is the standard hop in between. Also picks up FIX 2's reconstructed
  // vertex as a byproduct, since it's the same PFParticle loop.
  std::vector<art::Ptr<recob::PFParticle>> pfpVec;
  std::unordered_map<unsigned int, int> hitKeyToPfpIdx;      // hit key -> index into pfpVec
  std::unordered_map<unsigned int, int> hitKeyToClusterKey;  // hit key -> cluster.key()
  bool haveVtx = false;
  float vtxX = 0, vtxY = 0, vtxZ = 0;
  {
    art::Handle<std::vector<recob::PFParticle>> pfpHandle;
    if (e.getByLabel(fPandoraLabel, pfpHandle) && pfpHandle.isValid()) {
      art::fill_ptr_vector(pfpVec, pfpHandle);
      art::FindManyP<recob::Cluster> pfpClusterAssns(pfpHandle, e, fPandoraLabel);
      art::FindOneP<recob::Vertex> pfpVertexAssns(pfpHandle, e, fPandoraLabel);
      auto clusterHandle = e.getHandle<std::vector<recob::Cluster>>(fPandoraLabel);
      if (clusterHandle.isValid()) {
        art::FindManyP<recob::Hit> clusterHitAssns(clusterHandle, e, fPandoraLabel);
        for (size_t ip = 0; ip < pfpVec.size(); ++ip) {
          for (auto const& cl : pfpClusterAssns.at(ip)) {
            for (auto const& h : clusterHitAssns.at(cl.key())) {
              hitKeyToPfpIdx[h.key()] = ip;
              hitKeyToClusterKey[h.key()] = int(cl.key());
            }
          }
          // FIX 2: take the first primary PFP with a valid reconstructed
          // vertex as "the" event vertex. See the file-header note on the
          // multi-slice-primary caveat.
          if (!haveVtx && pfpVec[ip]->IsPrimary()) {
            art::Ptr<recob::Vertex> const& vtx = pfpVertexAssns.at(ip);
            if (!vtx.isNull()) {
              double xyz[3];
              vtx->XYZ(xyz);
              vtxX = xyz[0]; vtxY = xyz[1]; vtxZ = xyz[2];
              haveVtx = true;
            }
          }
        }
      }
      else {
        mf::LogWarning("ICARUSNuGraphAnalyzerRecoFix")
          << "no Cluster collection at " << fPandoraLabel
          << " -- icluster/ipfp will be -1 for all hits this event";
      }
    }
  }

  // FIX 2: actually fill and Fill() the event tree.
  _event  = e.event();
  _subrun = e.subRun();
  _run    = e.run();
  if (haveVtx) {
    _vtx_x = vtxX; _vtx_y = vtxY; _vtx_z = vtxZ;
  }
  else {
    _vtx_x = _vtx_y = _vtx_z = std::numeric_limits<float>::quiet_NaN();
  }
  _treeEvt->Fill();

  art::InputTag fNGFilterLabel{fNGLabel, "filter"};
  art::InputTag fNGSemanticLabel{fNGLabel, "semantic"};

  art::FindOneP<anab::FeatureVector<1>> find1FilterFromHit(allHits, e, fNGFilterLabel);
  art::FindOneP<anab::FeatureVector<5>> find1SemanticFromHit(allHits, e, fNGSemanticLabel);

  for (size_t hitIdx = 0; hitIdx < allHits.size(); hitIdx++) {
    art::Ptr<recob::Hit> hit = allHits[hitIdx];

    // Fill event information. (unchanged from original)
    _event  = e.event();
    _subrun = e.subRun();
    _run    = e.run();
    _id     = hit.key();

    // Fill slice ID. (unchanged from original)
    auto itSlice = hitToSliceID.find(hit.key());
    _islc = (itSlice != hitToSliceID.end()) ? itSlice->second : -1;

    // FIX 1: real lookups instead of leaving _icluster/_ipfp untouched.
    auto pfpIt = hitKeyToPfpIdx.find(hit.key());
    _ipfp = (pfpIt != hitKeyToPfpIdx.end()) ? int(pfpVec[pfpIt->second]->Self()) : -1;
    auto clIt = hitKeyToClusterKey.find(hit.key());
    _icluster = (clIt != hitKeyToClusterKey.end()) ? clIt->second : -1;

    const art::Ptr<anab::FeatureVector<1>> filter = find1FilterFromHit.at(hitIdx);
    const art::Ptr<anab::FeatureVector<5>> semantic = find1SemanticFromHit.at(hitIdx);

    if (filter.isNull() || semantic.isNull()) {
      std::cout << "Hit #" << hitIdx << " with key " << hit.key() << " had no valid NuGraphOutput. Skipping." << "\n";

      // FIX 3: explicit reset -- these are class members, so without this a
      // row for a hit with no NG output would silently carry the PREVIOUS
      // hit's real scores.
      _has_ng_output = 0;
      _x_filter = _MIP = _HIP = _shower = _michel = _diffuse = -1.;

      // Fill hit description. (unchanged from original)
      _wire  = hit->WireID().Wire;
      _plane = hit->WireID().Plane;
      _tpc   = hit->WireID().TPC;
      _cryo  = hit->WireID().Cryostat;
      _time  = hit->PeakTime();

      _treeHit->Fill();
      continue;
    }
    else {
      _has_ng_output = 1; // FIX 3
      // Fill NuGraph predictions. (unchanged from original)
      _x_filter = filter->at(0);
      _MIP      = semantic->at(0);
      _HIP      = semantic->at(1);
      _shower   = semantic->at(2);
      _michel   = semantic->at(3);
      _diffuse  = semantic->at(4);

      // Fill hit description. (unchanged from original)
      _wire  = hit->WireID().Wire;
      _plane = hit->WireID().Plane;
      _tpc   = hit->WireID().TPC;
      _cryo  = hit->WireID().Cryostat;
      _time  = hit->PeakTime();

      _treeHit->Fill();
    }
  }
}

DEFINE_ART_MODULE(ICARUSNuGraphAnalyzerRecoFix)
