from pathlib import Path
import pytest

from djura.vulnerability_modeller.vm_mdof import VMMDOF


path = Path(__file__).resolve().parent


@pytest.mark.skip("Deprecated")
class DeprecatedTestVM:
    mdof_defaults = path / "assets/vm/input.yaml"
    mdof_export = path / "assets/vm/mdof.json"
    backbone_path = path / "assets/vm/backbone.json"
    slf = path / "assets/vm/slf.json"

    def test_run(self):
        vm = VMMDOF(self.mdof_defaults)
        vm.define_capacity_curve(self.backbone_path)
        vm.predict_strength_ratio()
        vm.estimate_drifts()
        vm.estimate_pfas()
        vm.predict_collapse_capacity(im_max=10., steps=201)
        vm.read_slfs(self.slf)
        vm.vulnerability()
        vm.compute_eal()
        # vm.export_data(self.mdof_export)

        for i, c in enumerate(vm.data['cases']):
            print(f"Case {i} EALs: {c['eals']}")

    def test_backbone(self):
        vm = VMMDOF(self.mdof_defaults)
        vm.define_capacity_curve(self.backbone_path)
        vm.predict_strength_ratio()
        vm.predict_collapse_capacity(im_max=10., steps=201)
        vm.read_slfs(self.slf)

        print(vm.data['cases'][0].keys())

    @pytest.mark.skip(reason="local test")
    def test_temp(self):
        vm = VMMDOF(self.mdof_defaults)
        vm.define_capacity_curve(self.backbone_path)
        vm.predict_strength_ratio()
        vm.estimate_drifts()
        vm.estimate_pfas()
        vm.predict_collapse_capacity(im_max=10., steps=201)
        vm.read_slfs(self.slf)
        vm.vulnerability()
        vm.compute_eal()

        for i, c in enumerate(vm.data['cases']):
            print(f"Case {i} EALs: {c['eals']}")
